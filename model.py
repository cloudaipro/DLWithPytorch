import torch
from torch import nn as nn
import math
from util.unet import UNet

class UNetWrapper(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()

    def forward(self, input_batch):
        bn_output = self.input_batchnorm(input_batch)
        un_output = self.unet(bn_output)
        fn_output = self.final(un_output)
        return fn_output


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, conv_channels,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(conv_channels, conv_channels,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.maxpool(out)
        return out
        

class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        # Tail
        self.tail_bachnorm = nn.BatchNorm3d(1)
        
        # Backbone
        self.block1 = LunaBlock(1, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)
    
        # Header
        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def forward(self, x):
        out = self.tail_bachnorm(x)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        conv_flat = out.view(out.size(0), -1)
        # conv_flat = out.view(out.numel(), -1)
        out = self.head_linear(conv_flat)

        return out, self.head_softmax(out)
    
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d
            }:
                nn.init.kaiming_normal_(m.weight.data,
                                        a=0,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)
