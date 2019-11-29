import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from helper import LOG

__all__ = ['Inception3', 'inception_v3']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


class Inception3(nn.Module):

    def __init__(self, num_categories, add_intermediate_layers, num_outputs=1, num_classes=1000, aux_logits=True, transform_input=False):
        super(Inception3, self).__init__()

        self.intermediate_CLF = []
        self.add_intermediate_layers = add_intermediate_layers
        self.num_categories = num_categories
        self.num_outputs = num_outputs
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        if self.add_intermediate_layers == 2:

            self.intermediate_CLF.append(IntermediateClassifier(35, 256, self.num_categories))
            self.num_outputs += 1

            self.intermediate_CLF.append(IntermediateClassifier(35, 288, self.num_categories))
            self.num_outputs += 1

            self.intermediate_CLF.append(IntermediateClassifier(35, 288, self.num_categories))
            self.num_outputs += 1

            self.intermediate_CLF.append(IntermediateClassifier(17, 768, self.num_categories))
            self.num_outputs += 1

            self.intermediate_CLF.append(IntermediateClassifier(17, 768, self.num_categories))
            self.num_outputs += 1

            self.intermediate_CLF.append(IntermediateClassifier(17, 768, self.num_categories))
            self.num_outputs += 1

            self.intermediate_CLF.append(IntermediateClassifier(17, 768, self.num_categories))
            self.num_outputs += 1


        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
            # add 1 again, since there is an aux classifier
            self.num_outputs += 1

        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)

        if self.add_intermediate_layers == 2:
            self.intermediate_CLF.append(IntermediateClassifier(8, 1280, self.num_categories))
            self.num_outputs += 1

            self.intermediate_CLF.append(IntermediateClassifier(8, 2048, self.num_categories))
            self.num_outputs += 1

        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        intermediate_outputs = []

        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # print("Mixed_5b size: ", x.size())
        if self.add_intermediate_layers == 2:
            intermediate_outputs.append(self.intermediate_CLF[0](x))
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        if self.add_intermediate_layers == 2:
            intermediate_outputs.append(self.intermediate_CLF[1](x))
        # print("Mixed_5c size: ", x.size())
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        if self.add_intermediate_layers == 2:
            intermediate_outputs.append(self.intermediate_CLF[2](x))
        # print("Mixed_5d size: ", x.size())
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        if self.add_intermediate_layers == 2:
            intermediate_outputs.append(self.intermediate_CLF[3](x))
        # print("Mixed_6a size: ", x.size())
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        if self.add_intermediate_layers == 2:
            intermediate_outputs.append(self.intermediate_CLF[4](x))
        # print("Mixed_6b size: ", x.size())
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        if self.add_intermediate_layers == 2:
            intermediate_outputs.append(self.intermediate_CLF[5](x))
        # print("Mixed_6c size: ", x.size())
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        if self.add_intermediate_layers == 2:
            intermediate_outputs.append(self.intermediate_CLF[6](x))

        x = self.Mixed_6e(x)
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)

        x = self.Mixed_7a(x)
        if self.add_intermediate_layers == 2:
            intermediate_outputs.append(self.intermediate_CLF[7](x))

        x = self.Mixed_7b(x)
        if self.add_intermediate_layers == 2:
            intermediate_outputs.append(self.intermediate_CLF[8](x))

        x = self.Mixed_7c(x)

        x = F.avg_pool2d(x, kernel_size=8)
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            # print("intermediate_outputs + aux + x:,  ", len(intermediate_outputs)+1+1)
            return  intermediate_outputs[:7] + [aux] + intermediate_outputs[7:]+ [x]
        else:
            return intermediate_outputs + [x]


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)
        # self.intermediate_CLF.append(IntermediateClassifier(256, self.label_classes))
        # self.num_outputs += 1
    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class IntermediateClassifier(nn.Module):

    def __init__(self, global_pooling_size, num_channels, num_classes):
        """
        Classifier of a cifar10/100 image.

        :param num_channels: Number of input channels to the classifier
        :param num_classes: Number of classes to classify
        """
        super(IntermediateClassifier, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        # self.residual_block_type = residual_block_type
        self.device = 'cuda'
        # if self.residual_block_type == 2: # basicblock type, ResNet-18, ResNet-34
        #     kernel_size = int(3584/self.num_channels)
        # elif self.residual_block_type == 3: # bottleneck block, ResNet-50, ResNet-101, ResNet-152
        #     kernel_size = int(14336/self.num_channels)
        # else:
        #     NotImplementedError

        kernel_size = global_pooling_size

        print("kernel_size for global pooling: ", kernel_size)

        self.features = nn.Sequential(
            nn.AvgPool2d(kernel_size=(kernel_size, kernel_size)),
            nn.Dropout(p=0.2, inplace=False)
        ).to(self.device)
        # print("num_channels: ", num_channels, "\n")
        self.classifier = torch.nn.Sequential(
            nn.Linear(num_channels, num_classes)
        ).to(self.device)

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


def Elastic_InceptionV3(args, logfile):
    num_classes = args.num_classes
    add_intermediate_layers = args.add_intermediate_layers
    pretrained_weight = args.pretrained_weight

    model = Inception3(num_classes, add_intermediate_layers, aux_logits=True)

    if pretrained_weight == 1:
        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
        LOG("loaded ImageNet pretrained weights", logfile)

    elif pretrained_weight == 0:
        LOG("not loading ImageNet pretrained weights", logfile)

    else:
        LOG("parameter--pretrained_weight, should be 0 or 1", logfile)
        NotImplementedError

    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)

    for param in model.parameters():
        param.requires_grad = True
    LOG("*********************set all parameters as trainable, all params are requires_grad == True*********************", logfile)

    if add_intermediate_layers == 2:
        LOG("add intermediate layer classifiers", logfile)

        # get all extra classifiers params and final classifier params
        for inter_clf in model.intermediate_CLF:
            for param in inter_clf.parameters():
                param.requires_grad = True

    elif add_intermediate_layers == 0:
        LOG("not adding any intermediate layer classifiers", logfile)
    else:
        NotImplementedError

    for param in model.fc.parameters():
        param.requires_grad = True

    return model

