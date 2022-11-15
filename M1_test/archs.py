import torch
from torch import nn

__all__ = ['UNet']

from s_location import m_location
from secret_key import generater_key


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#scale_factor:放大的倍数  插值

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        A=[]
        x0_0 = self.conv0_0(input)
        a0_0 = m_location(x0_0)
        A.append(a0_0)
        A.append(a0_0-1)
        x1_0 = self.conv1_0(self.pool(x0_0))
        a1_0 = m_location(x1_0)
        A.append(a1_0)
        x2_0 = self.conv2_0(self.pool(x1_0))
        a2_0 = m_location(x2_0)
        A.append(a2_0)
        x3_0 = self.conv3_0(self.pool(x2_0))
        a3_0 = m_location(x3_0)
        A.append(a3_0)
        A.append(a3_0 - 1)
        x4_0 = self.conv4_0(self.pool(x3_0))
        a4_0 = m_location(x4_0)
        A.append(a4_0)
        A.append(a4_0 - 1)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        a3_1 = m_location(x3_1)
        A.append(a3_1)
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        a2_2 = m_location(x2_2)
        A.append(a2_2)
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        a1_3 = m_location(x1_3)
        A.append(a1_3)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        a0_4 = m_location(x0_4)
        A.append(a0_4)
        #print(A)

        output = self.final(x0_4)
        output = generater_key(output, A)
        return output
        A = clean_list(A)


