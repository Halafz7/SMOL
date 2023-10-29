import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from ops import *

def conv(in_planes, out_planes, kernel_size, stride, p_init, fp):
    padding = (kernel_size-1)//2
    if fp: return nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=False)
    return NoisyConv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=False, p_init=p_init)

def linear(in_feats, out_feats, p_init, fp):
    if fp: return nn.Linear(in_feats, out_feats)
    return NoisyLinear(in_feats, out_feats, p_init=p_init)

class ShortcutP(nn.Module):
  def __init__(self, inplanes, planes, p_init, fp):
    super(ShortcutP, self).__init__()
    self.avg = nn.AvgPool2d(2)

  def forward(self, x):
    x = self.avg(x)
    return torch.cat((x, x.mul(0)), 1)

class ShortcutC(nn.Module):
  def __init__(self, inplanes, planes, p_init, fp):
    super(ShortcutC, self).__init__()
    self.conv = conv(inplanes, planes, 1, 2, p_init, fp)

  def forward(self, x):
    return self.conv(x)

class ShortcutCB(nn.Module):
  def __init__(self, inplanes, planes, p_init, fp):
    super(ShortcutCB, self).__init__()
    self.conv = conv(inplanes, planes, 1, 2, p_init, fp)
    self.bn = nn.BatchNorm2d(planes)

  def forward(self, x):
    return self.bn(self.conv(x))

class NoisyConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bias, p_init, groups=1):
        super(NoisyConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.mode = 'noisy'
        self.qf = torch.floor
        self.groups = groups

        s = math.log(2**(1-p_init) / (1 - 2**(1-p_init)))
        self.weight_s = nn.Parameter(torch.ones_like(self.weight)*s)
        if self.bias is not None:
            self.bias_s = nn.Parameter(torch.ones_like(self.bias)*s)
        self.p, self.bp = None, None

    def forward(self, x):
        bias = None

        if self.mode == 'quant':
            weight, self.p = Quantize.apply(self.weight, self.weight_s, self.qf)
            if self.bias is not None:
                bias, self.bp = Quantize.apply(self.bias, self.bias_s, self.qf)
        elif self.mode == 'noisy':
            weight = self.weight + torch.sigmoid(self.weight_s) * torch.empty_like(self.weight).uniform_(-1, 1)
            if self.bias is not None:
                bias = self.bias + torch.sigmoid(self.bias_s) * torch.empty_like(self.bias).uniform_(-1, 1)

        out = F.conv2d(x, weight, bias=bias, stride=self.stride, padding=self.padding, groups=self.groups)
        return out

class NoisyLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, p_init):
        super(NoisyLinear, self).__init__(in_channels, out_channels)
        self.mode = 'noisy'
        self.qf = torch.floor

        s = math.log(2**(1-p_init) / (1 - 2**(1-p_init)))
        self.weight_s = nn.Parameter(torch.ones_like(self.weight)*s)
        if self.bias is not None:
            self.bias_s = nn.Parameter(torch.ones_like(self.bias)*s)
        self.p, self.bp = None, None

    def forward(self, x):
        bias = None

        if self.mode == 'quant':
            weight, self.p = Quantize.apply(self.weight, self.weight_s, self.qf)
            if self.bias is not None:
                bias, self.bp = Quantize.apply(self.bias, self.bias_s, self.qf)
        elif self.mode == 'noisy':
            weight = self.weight + torch.sigmoid(self.weight_s) * torch.empty_like(self.weight).uniform_(-1, 1)
            if self.bias is not None:
                bias = self.bias + torch.sigmoid(self.bias_s) * torch.empty_like(self.bias).uniform_(-1, 1)

        out = F.linear(x, weight, bias=bias)
        return out

