from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import os
import os.path
import time
import random
import torch
import torchvision
import torch.nn as nn
from torch.nn import init
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import scipy
import scipy.stats as st
import scipy.io as scio
from collections import OrderedDict
import math
import matplotlib.pyplot as plt
from scipy.io import savemat
from skimage.measure import profile_line

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, up_mode='transpose', scaling=True, conv_enable=True):
        super(Up, self).__init__()
        self.in_channels=in_channels
        self.conv_enable=conv_enable
        if scaling:
          if up_mode == 'transpose':
              self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
              self.conv = DoubleConv2d(in_channels, out_channels)

          else:
              self.up = nn.Upsample(mode='bilinear', scale_factor=2)
              self.conv = DoubleConv2d(in_channels, out_channels, in_channels//2)
        else:
          self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
          self.conv = DoubleConv2d(in_channels, out_channels)


    def forward(self, from_up, from_skip):
        from_up = self.up(from_up)
        x = torch.cat([from_skip, from_up], dim=1)
        return self.conv(x) if self.conv_enable else x 


class Down(nn.Module):
    """ Downscaling with maxpool then double conv """
    def __init__(self,in_channels,out_channels, pooling=True):
        super(Down, self).__init__()
        if pooling:
          self.down = nn.Sequential(
              nn.MaxPool2d(kernel_size=2, stride=2),
              DoubleConv2d(in_channels, out_channels)
          )
        else:
          self.down = nn.Sequential(
              DoubleConv2d(in_channels, out_channels)
          )
    def forward(self, x):
        return self.down(x)


class OutConv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(OutConv,self).__init__()
        self.conv = conv1x1(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)



class DoubleConv2d(nn.Module):
    """(Convolution => BN => ReLU) * 2 """
    def __init__(self,in_channels, out_channels, mid_channels=None, BN=False, kernel_size=3, dilation=1):
        super(DoubleConv2d,self).__init__()
        if not mid_channels:
            self.mid_channels = out_channels
        else:
            self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        if BN:
          self.double_conv = nn.Sequential(
              nn.Conv2d(self.in_channels, self.mid_channels,kernel_size=kernel_size,stride=1,dilation=self.dilation,padding=self.dilation,bias=True,groups=1),
              nn.BatchNorm2d(self.mid_channels),
              nn.ReLU(),
              nn.Conv2d(self.mid_channels, self.out_channels,kernel_size=kernel_size,stride=1,dilation=self.dilation,padding=self.dilation,bias=True,groups=1),
              nn.BatchNorm2d(self.out_channels),
              nn.ReLU(),
          )
        else:
          self.double_conv = nn.Sequential(
              nn.Conv2d(self.in_channels, self.mid_channels,kernel_size=kernel_size,stride=1,dilation=self.dilation,padding=self.dilation,bias=True,groups=1),
              nn.ReLU(),
              nn.Conv2d(self.mid_channels, self.out_channels,kernel_size=kernel_size,stride=1,dilation=self.dilation,padding=self.dilation,bias=True,groups=1),
              nn.ReLU(),
          )
    def forward(self,x):
        return self.double_conv(x)


class MLP(nn.Module):

    def __init__(self,in_channels, out_channels, mid_channels=None):
        super(MLP,self).__init__()
        if not mid_channels:
            self.mid_channels = out_channels
        else:
            self.mid_channels = mid_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.mid_channels,kernel_size=1,stride=1,padding=0,bias=True,groups=1),
            nn.ReLU(),
            nn.Conv2d(self.mid_channels, self.out_channels,kernel_size=1,stride=1,padding=0,bias=True,groups=1),
            nn.ReLU(),
        )

    def forward(self,x):
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(OutConv,self).__init__()
        self.conv = conv1x1(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)


class cAM(nn.Module):
    def __init__(self, in_channels, ratio=4, softmax=True):
        super(cAM, self).__init__()
        self.softmax = softmax
        self.in_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # [N, C, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool2d(1) # [N, C, 1, 1]

        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
                     nn.ReLU(),
                     nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out 
        if self.softmax:
          out = F.softmax(x, dim = 1) * (0.25 * self.in_channels)
        else:
          out = self.sigmoid(out)
        return out


class sAM1(nn.Module): # For dataset RS, AR, and EXP 
    def __init__(self, mask, kernel_size=5, softmax=True):
        super(sAM1, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = softmax
        self.mask = mask

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        x = x * self.mask # clear non-medium pixel value
        x = x + ((1 - self.mask) * -1e8) # eliminate affect of non-medium pixel in softmax activiation fuction

        if self.softmax:
          x_size = x.shape
          x = torch.reshape(x,(x_size[0],1,-1))
          x = F.softmax(x, dim = 2)
          x = torch.reshape(x,x_size)*(x_size[2]*x_size[3] * 0.5) # baseline: 0.5
        else:
          x = self.sigmoid(x)
        return x


class EAPNet1(nn.Module): # For dataset RS, AR, and EXP 

    def __init__(self, mask, in_channels=1, up_mode='transpose', pooling = True):

        super(EAPNet1, self).__init__()

        self.up_mode = up_mode

        self.in_channels = in_channels
        self.pooling = pooling
        self.scaling = pooling

        self.mask = mask

        self.inconv = DoubleConv2d(1, 32)
        self.down1 = Down(32, 64, pooling=self.pooling)
        self.down2 = Down(64, 128, pooling=self.pooling)
        self.down3 = Down(128, 256, pooling=self.pooling)

        self.up1 = Up(256, 128, scaling=self.scaling)
        self.up2 = Up(128, 64, scaling=self.scaling)
        self.up3 = Up(64, 32, scaling=self.scaling)
        self.ca = cAM(32) # unused in EAPNet
        self.sa = sAM1(self.mask, softmax=True)
        self.outconv1 = MLP(32, 16, 16)
        self.outconv2 = OutConv(16, 1)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
              init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, PAimg):

        # encoder1: PAimg
        e1 = self.inconv(PAimg)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        out = self.down3(e3)

        # upsample path
        out = self.up1(out,e3)
        out = self.up2(out,e2)
        out = self.up3(out,e1) 

        out = self.sa(out)*out

        out = self.outconv1(out)
        out = self.outconv2(out)

        return out


class sAM2(nn.Module):
    # For dataset V and ST
    def __init__(self, kernel_size=5, softmax=True):
        super(sAM2, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = softmax

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x) 
        if self.softmax:
          x_size = x.shape
          x = torch.reshape(x,(x_size[0],1,-1))
          x = F.softmax(x, dim = 2)
          x = torch.reshape(x,x_size)*(x_size[2]*x_size[3]*0.5)
        else:
          x = self.sigmoid(x)
        return x


class EAPNet2(nn.Module):
    # For dataset V and ST

    def __init__(self, in_channels=1, up_mode='transpose', pooling = True):

        super(EAPNet2, self).__init__()

        self.up_mode = up_mode
        self.in_channels = in_channels
        self.pooling = pooling
        self.scaling = pooling

        self.inconv = DoubleConv2d(1, 32)
        self.down1 = Down(32, 64, pooling=self.pooling)
        self.down2 = Down(64, 128, pooling=self.pooling)
        self.down3 = Down(128, 256, pooling=self.pooling)

        self.up1 = Up(256, 128, scaling=self.scaling)
        self.up2 = Up(128, 64, scaling=self.scaling)
        self.up3 = Up(64, 32, scaling=self.scaling)
        self.ca = cAM(32) # unused in EAPNet
        self.sa = sAM2(softmax=True)
        self.outconv1 = MLP(32, 16, 16)
        self.outconv2 = OutConv(16, 1)
        # 初始化权重函数
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
              init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, PAimg):
        # encoder1: PAimg
        e1 = self.inconv(PAimg)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        out = self.down3(e3)

        # upsample path
        out = self.up1(out,e3)
        out = self.up2(out,e2)
        out = self.up3(out,e1)

        out = self.sa(out)*out

        out = self.outconv1(out)
        out = self.outconv2(out)
        return out