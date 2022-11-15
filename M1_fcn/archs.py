import torch
from torch import nn

__all__ = ['FCNs']

from torchvision.models import vgg16
from torch import nn
from torchvision.models import vgg16

#####FCN.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG

from s_location import m_location
from secret_key import generater_key


class FCNs(nn.Module):#定义FCN网络框架

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net#初始化，pretrained_net加载预训练模型，定义分类个数
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)#该层采用1*1卷积,目的为不改变长宽，而改变通道数为给出的类别数

    def forward(self, x):#x代表原始输入图片
        output = self.pretrained_net(x)
        A = []
        x4 = output['x4']
        a4 = m_location(x4)
        A.append(a4)
        A.append(a4%96)
        x3 = output['x3']
        a3 = m_location(x3)
        A.append(a3)
        A.append(a3 %96)
        x2 = output['x2']
        a2 = m_location(x2)
        A.append(a2)
        A.append(a2 %96)
        x1 = output['x1']
        a1 = m_location(x1)
        A.append(a1)
        A.append(a1%96)
        score = self.bn2(self.relu(self.deconv2(x4)))
        score = score + x3
        a4_1 = m_location(score)
        A.append(a4_1)
        score = self.bn3(self.relu(self.deconv3(score)))
        score = score + x2
        a4_2 = m_location(score)
        A.append(a4_2)
        score = self.bn4(self.relu(self.deconv4(score)))
        score = score + x1
        a4_3 = m_location(score)
        A.append(a4_3)
        score = self.bn5(self.relu(self.deconv5(score)))
        a4_4 = m_location(score)
        A.append(a4_4)
        score = self.classifier(score)
        #print(A)
        score = generater_key(score, A)
        return score

        A = clean_list(A)


class VGGNet(VGG):
    def __init__(self, pretrained=False, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=False).state_dict())" % model)
        #requires_grad——是否保存梯度信息，决定了是否能进行反向传播，这里选择的方案为只导入网络结构，不导入参数，所以pretrained设置为False，同时，给出不保存梯度信息的选择

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        # delete redundant fully-connected layer params, can save memory
        # 去掉vgg最后的全连接层(classifier)
        if remove_fc:
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        # 获取每个maxpooling层的输出（vggnet中有5个maxpool）
        # 因为FCN类的使用在每一个池化层结束后中断一次，得到一个output，由于vgg中有五个池化层，所以最后output字典中会有五个output结构
        for idx, (begin, end) in enumerate(self.ranges):
            # self.ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)) (vgg16 examples)
            for layer in range(begin, end):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# Vgg-Net config
# Vgg网络结构配置
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# make layers using Vgg-Net config(cfg)
# # 由cfg构建vgg-Net 详情见vggnet模型搭建
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


'''
VGG-16网络参数
Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace)
  (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU(inplace)
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (6): ReLU(inplace)
  (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (8): ReLU(inplace)
  (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): ReLU(inplace)
  (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (13): ReLU(inplace)
  (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (15): ReLU(inplace)
  (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (18): ReLU(inplace)
  (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (20): ReLU(inplace)
  (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (22): ReLU(inplace)
  (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (25): ReLU(inplace)
  (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (27): ReLU(inplace)
  (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (29): ReLU(inplace)
  (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
'''

if __name__ == "__main__":
    pass



