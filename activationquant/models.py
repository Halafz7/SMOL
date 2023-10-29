import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from layers import *

class MixedNet(nn.Module):
    def __init__(self, mode, qf, p_init):
        super(MixedNet, self).__init__()

        self.mixed_layers = []
        self.mode = mode
        self.qf = qf
        self.p_init = p_init

    def set_mode(self, mode):
        self.mode = mode
        for m in self.mixed_layers: m.mode = mode

    def set_qf(self, qf):
        self.qf = qf
        for m in self.mixed_layers: m.qf = qf

    def prec_cost(self):
        cost = 0
        for m in self.mixed_layers:
            if m.s is not None:
                cost += - torch.log2(torch.sigmoid(m.s)).sum()
        return cost

    def print_precs(self):
        precs = 0
        num_precs = 0
        for m in self.mixed_layers:
            if m.p is not None:
                precs += m.p.sum().item()
                num_precs += m.p.numel()
            else:
                p_float = 1-torch.log2(torch.sigmoid(m.s))
                precs += self.qf(p_float).sum().item()
                num_precs += p_float.numel()

        if num_precs == 0: return "Network is in full precision"
        return "Network precision with {}: {}".format(self.qf.__name__, precs/num_precs)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, precision, qact, alpha, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = qact((1,planes,32//(planes//16),32//(planes//16)), precision, alpha)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = qact((1,planes,32//(planes//16),32//(planes//16)), precision, alpha)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x:
                                        F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

    def forward(self, x):
        residual = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None: residual = self.shortcut(x)
        out += residual
        out = self.act2(out)
        return out

class ResNet(MixedNet):
    def __init__(self, p_init, qact, alpha, mode='quant', qf=torch.floor):
        super(ResNet, self).__init__(mode, qf, p_init)
        self.in_planes = 16
        self.p_init = p_init
        self.alpha_init = alpha
        self.qact = {'pact' : PACT, 'repact' : REPACT, 'pelt' : PELT, 'pelt2' : PELT2}[qact]
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act = self.qact((1,16,32,32), self.p_init, alpha)
        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)
        self.linear = nn.Linear(64, 10)

        self.mixed_layers = [m for m in self.modules() if True in [isinstance(m, c) for c in qamodules]]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.p_init, self.qact, self.alpha_init, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
