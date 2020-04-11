import numpy as np
import scipy.signal
import torch
from math import ceil
from torch import nn as nn
from torch.nn import Parameter, functional as F
from torch.nn.init import xavier_normal
from torch.nn.modules.utils import _pair
from torch.utils.data import DataLoader
from tqdm import tqdm


def elu1(x):
    return F.elu(x, inplace=True) + 1.0


class Elu1(nn.Module):
    """
    Elu activation function shifted by 1 to ensure that the
    output stays positive. That is:
    Elu1(x) = Elu(x) + 1
    """

    def forward(self, x):
        return elu1(x)


def log1exp(x):
    return torch.log(1.0 + torch.exp(x))


class Log1Exp(nn.Module):
    def forward(self, x):
        return log1exp(x)


class DepthSeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.add_module("in_depth_conv", nn.Conv2d(in_channels, out_channels, 1, bias=bias))
        self.add_module(
            "spatial_conv",
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                bias=bias,
                groups=out_channels,
            ),
        )
        self.add_module("out_depth_conv", nn.Conv2d(out_channels, out_channels, 1, bias=bias))
