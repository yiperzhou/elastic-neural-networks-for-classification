from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
import sys
sys.path.append("../")

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from helper import LOG

__all__ = ['Elastic_ResNet']

# global num_outputs
# initially only one classifier output

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
# ===========================================================================below residual network source code ================
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class IntermediateClassifier(nn.Module):

    def __init__(self, num_channels, residual_block_type, num_classes):
        """
        Classifier of a cifar10/100 image.

        :param num_channels: Number of input channels to the classifier
        :param num_classes: Number of classes to classify
        """
        super(IntermediateClassifier, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.residual_block_type = residual_block_type
        self.device = 'cuda'
        if self.residual_block_type == 2: # basicblock type, ResNet-18, ResNet-34, then feature_maps_width(height) * num_channels == 3584
            kernel_size = int(3584/self.num_channels)
        elif self.residual_block_type == 3: # bottleneck block, ResNet-50, ResNet-101, ResNet-152, then feature_maps_width(height) * num_channels == 14336
            kernel_size = int(14336/self.num_channels)
        else:
            NotImplementedError
            
        print("kernel_size for global pooling: ", kernel_size)

        self.features = nn.Sequential(
            nn.AvgPool2d(kernel_size=(kernel_size, kernel_size)),
            nn.Dropout(p=0.2, inplace=False)
        ).to(self.device)
        # print("num_channels: ", num_channels, "\n")
        # 在keras中这里还有dropout rate = 0.2，但是这里没有，需要添加一下
        self.classifier = torch.nn.Sequential(nn.Linear(self.num_channels, self.num_classes)).to(self.device)

    def forward(self, x):
        """
        Drive features to classification.

        :param x: Input of the lowest scale of the last layer of
                  the last block
        :return: Cifar object classification result
        """
        # get the width or heigh on that feaure map
        # kernel_size = x.size()[-1]
        # get the number of feature maps
        # num_channels = x.size()[-3]
        
        # print("kernel_size for global pooling: " ,kernel_size)
        

        # do global average pooling
        x = self.features(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, residual_block_type, cifar_classes, add_intermediate_layers, num_outputs=1, num_classes=1000):
        self.intermediate_CLF = []
        self.add_intermediate_layers = add_intermediate_layers
        self.cifar_classes = cifar_classes
        self.residual_block_type = residual_block_type
        self.num_outputs = num_outputs
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        # layers = [None] * (1+(blocks-1)*2) #自己添加的
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        
        if self.add_intermediate_layers == 2:
            # global num_outputs #using this variable to count the number of CLF
            self.intermediate_CLF.append(IntermediateClassifier(self.inplanes, self.residual_block_type, self.cifar_classes))
            self.num_outputs += 1

        # print("blocks: ", 1, "/", blocks, ", self.inplanes: ", self.inplanes, ", planes: ", planes)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            # print("blocks: ", i+1, "/", blocks, ", self.inplanes: ", self.inplanes, ", planes: ", planes)
            if self.add_intermediate_layers == 2:
                # global num_outputs
                if i == (blocks-1) and planes == 512:# means this is the intermediate classifier has close position with the final output classifier
                    # not append intermediate classifier
                    print("skip the last intermediate classifer, since this classifier has close position with the final output classifier")
                else:
                    self.intermediate_CLF.append(IntermediateClassifier(self.inplanes, self.residual_block_type, self.cifar_classes))
                    self.num_outputs += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        i = 0
        intermediate_outputs = []
        final_inter_clf_position = len(self.intermediate_CLF)
        # print("=====> # of intermediate classifiers: ", len(self.intermediate_CLF), ", total classifiers: ", len(self.intermediate_CLF)+1)
        
        # make sure insert an intermediate classifier after each residul block 
        # assert len(self.intermediate_CLF) == len(self.layer1)+len(self.layer2)+len(self.layer3)+len(self.layer4)

        for res_layer in self.layer1:
            x = res_layer(x)
            if self.add_intermediate_layers == 2:
                intermediate_outputs.append(self.intermediate_CLF[i](x))
                i += 1

        for res_layer in self.layer2:
            x = res_layer(x)
            if self.add_intermediate_layers == 2:
                intermediate_outputs.append(self.intermediate_CLF[i](x))
                i += 1

        for res_layer in self.layer3:
            x = res_layer(x)
            if self.add_intermediate_layers == 2:
                intermediate_outputs.append(self.intermediate_CLF[i](x))
                i += 1                    
        
        for res_layer in self.layer4:
            x = res_layer(x)
            if self.add_intermediate_layers == 2:
                if i == final_inter_clf_position:
                    # print("forward function, skip the final intermediate classifier")
                    pass
                else:
                    intermediate_outputs.append(self.intermediate_CLF[i](x))
                    i += 1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return intermediate_outputs+[x]


def Elastic_ResNet(args, logfile):
    
    # num_outputs = 1 # initially only one classifier output

    num_classes = args.num_classes
    add_intermediate_layers = args.add_intermediate_layers
    pretrained_weight = args.pretrained_weight

    model_weight_url = None
    if args.model == "Elastic_ResNet18":
        # residual block type, 2 is BasicBlock, which means 2 conv-bn-relu in one block, 3 is BottleneckBlock, which means 3 conv-bn-relu blocks
        residual_block_type = 2
        model = ResNet(BasicBlock, [2, 2, 2, 2], residual_block_type, num_classes, add_intermediate_layers)
        model_weight_url = model_urls['resnet18']
        LOG("successfully create model: (Elastic-)ResNet18", logfile)

    elif args.model == "Elastic_ResNet34":
        # residual block type, 2 is BasicBlock, which means 2 conv-bn-relu in one block, 3 is BottleneckBlock, which means 3 conv-bn-relu blocks
        residual_block_type = 2
        model = ResNet(BasicBlock, [3, 4, 6, 3], residual_block_type, num_classes, add_intermediate_layers) 
        model_weight_url =  model_urls['resnet34']    
        LOG("successfully create model: (Elastic-)ResNet34", logfile)  

    elif args.model == "Elastic_ResNet50":
        residual_block_type = 3
        model = ResNet(Bottleneck, [3, 4, 6, 3], residual_block_type, num_classes, add_intermediate_layers)   
        model_weight_url = model_urls['resnet50']  
        LOG("successfully create model: (Elastic-)ResNet50", logfile)

    elif args.model == "Elastic_ResNet101":
        residual_block_type = 3
        model = ResNet(Bottleneck, [3, 4, 23, 3], residual_block_type, num_classes, add_intermediate_layers)   
        model_weight_url = model_urls['resnet101']  
        LOG("successfully create model: (Elastic-)ResNet101", logfile)

    elif args.model == "Elastic_ResNet152":
        residual_block_type = 3
        model = ResNet(Bottleneck,  [3, 8, 36, 3], residual_block_type, num_classes, add_intermediate_layers)   
        model_weight_url = model_urls['resnet152']  
        LOG("successfully create model: (Elastic-)ResNet152", logfile)


    if pretrained_weight == 1:
        model.load_state_dict(model_zoo.load_url(model_weight_url))
        LOG("loaded ImageNet pretrained weights", logfile)
        
    elif pretrained_weight == 0:
        LOG("not loading ImageNet pretrained weights", logfile)

    else:
        LOG("parameter--pretrained_weight, should be 0 or 1", logfile)
        NotImplementedError

    # if add_intermediate_layers == 0: # not adding any intermediate layer classifiers
    #     print("not adding any intermediate layer classifiers")    
    #     LOG("not adding any intermediate layer classifiers", logfile)
    # elif add_intermediate_layers == 2:
    #     print("add any intermediate layer classifiers")    
    #     LOG("add intermediate layer classifiers", logfile)


    # print("=====> successfully load pretrained imagenet weight")
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)

    for param in model.parameters():
        param.requires_grad = False
    
    if add_intermediate_layers == 2:
        LOG("add intermediate layer classifiers", logfile)

        # get all extra classifiers params and final classifier params
        for inter_clf in model.intermediate_CLF:
            for param in inter_clf.parameters():
                param.requires_grad = True
        
        for param in model.fc.parameters():
            param.requires_grad = True 
    
    elif add_intermediate_layers == 0:
        LOG("not adding any intermediate layer classifiers", logfile)

        for param in model.fc.parameters():
            param.requires_grad = True         
    else:
        NotImplementedError

    return model
