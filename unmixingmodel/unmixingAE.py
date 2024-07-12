import math
import torch
import torch.nn as nn
import numpy as np
from core.common import *

class UnmixingAE(nn.Module):
    def __init__(self, n_blocks, res_scale, input_channels, conv= default_conv):
        super(UnmixingAE,self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        self.input_channels = input_channels

        self.layerup1 = conv(self.input_channels, 32,kernel_size)
        self.layerup2 = SSPN(32, n_blocks,64,act,res_scale)

        self.layerup3 = SSPN(64, n_blocks,128,act,res_scale)
        self.layer1 = SSPN(128, n_blocks,96,act,res_scale)
        self.layer2 = SSPN(96,n_blocks,48, act,res_scale)
        self.layer3 = SSPN(48,n_blocks,5, act,res_scale)	
        self.encodelayer = nn.Sequential(nn.Softmax())

        self.decoderlayer = nn.Conv2d(in_channels=5, out_channels=self.input_channels,kernel_size=(1,1),bias=False)

	
    def forward(self, x):
        unx = self.layerup1(x)
        unx = self.layerup2(unx)
        unx = self.layerup3(unx)
        unx = self.layer1(unx)
        unx = self.layer2(unx)
        unx = self.layer3(unx)
        un_result = self.encodelayer(unx)
        de_result = self.decoderlayer(un_result)
        decoder_weight = self.get_last_layer()
	
        return un_result, de_result,decoder_weight

    def get_last_layer(self):
        return self.decoderlayer.weight


class SSB(nn.Module):
    def __init__(self, n_feats, kernel_size, act,res_scale,conv=default_conv):
        super(SSB, self).__init__()
        self.spa = ResBlock(conv,
                            n_feats,
                            kernel_size,
                            act=act,
                            res_scale=res_scale)
        self.spc = ResAttentionBlock(conv,
                                     n_feats,
                                     1,
                                     act=act,
                                     res_scale=res_scale)

    def forward(self, x):
        return self.spc(self.spa(x))


class SSPN(nn.Module):
    def __init__(self, n_feats, n_blocks, num_channels , act,res_scale,conv=default_conv):
        super(SSPN, self).__init__()

        kernel_size = 3
        m = []

        for i in range(n_blocks):
            m.append(SSB(n_feats, kernel_size, act=act,res_scale = res_scale))

        self.finnallayer = conv(n_feats,num_channels, kernel_size)
        self.net = nn.Sequential(*m)

    def forward(self, x):
        res = self.net(x)
        res += x
        out = self.finnallayer(res)

        return out
