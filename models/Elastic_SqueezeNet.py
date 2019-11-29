from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
import sys
sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import math
from helper import LOG



__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, num_categories, add_intermediate_layers, num_outputs=1, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.intermediate_CLF = []
        self.add_intermediate_layers = add_intermediate_layers
        self.num_categories = num_categories
        self.num_outputs = num_outputs

        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),     
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
            if self.add_intermediate_layers == 2:                
                self.intermediate_CLF.append(IntermediateClassifier(54, 128, self.num_categories))
                self.num_outputs += 1     

                self.intermediate_CLF.append(IntermediateClassifier(54, 128, self.num_categories))
                self.num_outputs += 1

                self.intermediate_CLF.append(IntermediateClassifier(54, 256, self.num_categories))
                self.num_outputs += 1     
                
                self.intermediate_CLF.append(IntermediateClassifier(27, 256, self.num_categories))
                self.num_outputs += 1     

                self.intermediate_CLF.append(IntermediateClassifier(27, 384, self.num_categories))
                self.num_outputs += 1                     

                self.intermediate_CLF.append(IntermediateClassifier(27, 384, self.num_categories))
                self.num_outputs += 1        

                self.intermediate_CLF.append(IntermediateClassifier(27, 512, self.num_categories))
                self.num_outputs += 1     

        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # print("forward function")
        intermediate_outputs = []

        # x = self.features(x)
        if self.add_intermediate_layers == 2:
            x0 = self.features[:4](x)
            intermediate_outputs.append(self.intermediate_CLF[0](x0))   

            x1 = self.features[4](x0)
            intermediate_outputs.append(self.intermediate_CLF[1](x1))   

            x2 = self.features[5](x1)
            intermediate_outputs.append(self.intermediate_CLF[2](x2))   

            x3 = self.features[6:8](x2)
            intermediate_outputs.append(self.intermediate_CLF[3](x3))   

            x4 = self.features[8](x3)
            intermediate_outputs.append(self.intermediate_CLF[4](x4))   

            x5 = self.features[9](x4)
            intermediate_outputs.append(self.intermediate_CLF[5](x5))   

            x6 = self.features[10](x5)
            intermediate_outputs.append(self.intermediate_CLF[6](x6))   

            # print("x6 shape: ", x6.size())
            x7 = self.features[11:](x6)

        elif self.add_intermediate_layers == 0:
            x7 = self.features(x)

        x = self.classifier(x7)
        # 这里实际上应该是 x.view(x.size(0), -1)才对?
        return intermediate_outputs + [x.view(x.size(0), self.num_categories)]


class IntermediateClassifier(nn.Module):

    def __init__(self, global_pooling_size, num_channels, num_classes):
        """
        Classifier of a cifar10/100 image.

        :param num_channels: Number of input channels to the classifier
        :param num_classes: Number of classes to classify
        """
        super(IntermediateClassifier, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.device = 'cuda'
        
        kernel_size = global_pooling_size

        self.features = nn.Sequential(
            nn.AvgPool2d(kernel_size=(kernel_size, kernel_size)),
            nn.Dropout(p=0.2, inplace=False)
        ).to(self.device)

        self.classifier = torch.nn.Sequential(nn.Linear(num_channels, num_classes)).to(self.device)

    def forward(self, x):
        """
        Drive features to classification.

        :param x: Input of the lowest scale of the last layer of
                  the last block
        :return: Cifar object classification result
        """
        x = self.features(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



def squeezenet1_0(pretrained=False, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.0, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_0']))
    return model


def Elastic_SqueezeNet(args, logfile):
    num_categories = args.num_classes
    add_intermediate_layers = args.add_intermediate_layers
    pretrained_weight = args.pretrained_weight

    model = SqueezeNet(num_categories=num_categories, add_intermediate_layers=add_intermediate_layers, version=1.0)

    if pretrained_weight == 1:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_0']))    
        LOG("loaded ImageNet pretrained weights", logfile)
        
    elif pretrained_weight == 0:
        LOG("not loading ImageNet pretrained weights", logfile)

    else:
        LOG("parameter--pretrained_weight, should be 0 or 1", logfile)
        NotImplementedError


    model.classifier._modules["1"] = nn.Conv2d(512, num_categories, kernel_size=(1, 1))

    for param in model.parameters():
        param.requires_grad = False
    
    if add_intermediate_layers == 2:
        LOG("set all intermediate classifiers and final classifiers parameter as trainable.", logfile)
        # get all extra classifiers params and final classifier params
        for inter_clf in model.intermediate_CLF:
            for param in inter_clf.parameters():
                param.requires_grad = True

    elif add_intermediate_layers == 0:
        LOG("only set final classifiers parameter as trainable.", logfile)
    
    else:
        NotImplementedError

    for param in model.classifier.parameters():
        param.requires_grad = True     
    
    return model    