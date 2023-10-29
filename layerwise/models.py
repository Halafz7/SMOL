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
            cost += -torch.log2(torch.sigmoid(m.weight_s)) * m.weight.numel()
            if m.bias is not None: cost += - torch.log2(torch.sigmoid(m.bias_s)) * m.bias.numel()
        return cost

    def project(self):
        for m in self.mixed_layers:
            p = 1-torch.log2(torch.sigmoid(m.weight_s))
            M = 2-2**(1-p)
            m.weight.data = torch.minimum(m.weight.data, M)
            m.weight.data = torch.maximum(m.weight.data, -M)

            if m.bias is not None:
                p = 1-torch.log2(torch.sigmoid(m.bias_s))
                M = 2-2**(1-p)
                m.bias.data = torch.minimum(m.bias.data, M)
                m.bias.data = torch.maximum(m.bias.data, -M)

    def print_precs(self):
        precs = 0
        norm_precs = 0
        num_precs = 0
        prec_list = []
        for m in self.mixed_layers:
            if m.p is not None:
                prec_list.append(m.p.item())
                precs += m.p * m.weight.numel()
                norm_precs = precs
                num_precs += m.weight.numel()
            else:
                scaling = max(1e-6, min(1, torch.abs(m.weight).max().item()))
                noise_mag = torch.sigmoid(m.weight_s)

                p_float = 1-torch.log2(noise_mag)
                p_float = torch.clamp(p_float, min=1)

                norm_p_float = 1-torch.log2(noise_mag/scaling)
                norm_p_float = torch.clamp(norm_p_float, min=1)

                prec_list.append(self.qf(norm_p_float).item())
                precs += self.qf(p_float) * m.weight.numel()
                norm_precs += self.qf(norm_p_float) * m.weight.numel()
                num_precs += m.weight.numel()

            if m.bias is not None:
                if m.bp is not None:
                    precs += m.bp * m.bias.numel()
                    norm_precs = precs
                    num_precs += m.bias.numel()
                else:
                    scaling = max(1e-6, min(1, torch.abs(m.bias).max().item()))
                    noise_mag = torch.sigmoid(m.bias_s)

                    p_float = 1-torch.log2(noise_mag)
                    p_float = torch.clamp(p_float, min=1)

                    norm_p_float = 1-torch.log2(noise_mag/scaling)
                    norm_p_float = torch.clamp(norm_p_float, min=1)

                    precs += self.qf(p_float) * m.bias.numel()
                    norm_precs += self.qf(norm_p_float) * m.bias.numel()
                    num_precs += m.bias.numel()

        if num_precs == 0: return "Network is in full precision"
        return "Network precision with {}: {} / {}\n{}".format(self.qf.__name__, (precs/num_precs).item(), (norm_precs/num_precs).item(), prec_list)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, NoisyConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class PreActBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, shortcut, p_init, fp, first=False, prune=False):
        super(PreActBlock, self).__init__()
        self.prune = prune
        self.first = first
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv(inplanes, planes, 3, stride, p_init, fp)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv(planes, planes, 3, 1, p_init, fp)
        self.relu2 = nn.ReLU(inplace=True)
        self.shortcut = shortcut

        if self.prune:
            self.conv1.p = torch.tensor(0.0)
            self.conv2.p = torch.tensor(0.0)

    def forward(self, x):
        if self.prune: return x

        residual = x
        out = self.relu1(self.bn1(x))
        if self.first: residual = out
        out = self.relu2(self.bn2(self.conv1(out)))
        out = self.conv2(out)
        if self.shortcut is not None: residual = self.shortcut(residual)
        out += residual
        return out

class PreResNet(MixedNet):
    def __init__(self, mode='quant', qf=torch.floor, p_init=32, fp_layers='shortcuts', shortcut_type='CB', prune=0):
        super(PreResNet, self).__init__(mode, qf, p_init)
        self.p_init = p_init
        self.fp_layers = fp_layers
        self.shortcut_module = {'P':ShortcutP, 'C':ShortcutC, 'CB':ShortcutCB}[shortcut_type]

        fp_firstconv = (fp_layers=='ends' or fp_layers=='all')
        self.pre = conv(3, 16, 3, 1, p_init, fp_firstconv)
        self.stage1 = self.make_stage(16, 16, 3, 1)
        self.stage2 = self.make_stage(16, 32, 3, 2)
        self.stage3 = self.make_stage(32, 64, 3, 2, prune)
        self.post = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.AvgPool2d(8))

        fp_fc = (fp_layers == 'all' or fp_layers == 'ends')
        self.fc = linear(64, 10, p_init, fp_fc)

        self.mixed_layers = [m for m in self.modules() if isinstance(m, NoisyConv2d) or isinstance(m, NoisyLinear)]
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_stage(self, inplanes, planes, blocks, stride, prune=0):
        if prune == 0: prune = []
        elif prune == 1: prune = [1]
        elif prune == 2: prune = [1, 2]

        fp_shortcut = (self.fp_layers=='all' or self.fp_layers == 'shortcuts')
        fp_blocks = (self.fp_layers=='all')

        if stride != 1 or inplanes != planes: shortcut = self.shortcut_module(inplanes, planes, self.p_init, fp_shortcut)
        else: shortcut = None

        layers = [PreActBlock(inplanes, planes, stride, shortcut, self.p_init, fp_blocks)]
        for i in range(1, blocks):
            layers.append(PreActBlock(planes, planes, 1, None, self.p_init, fp_blocks, prune=(i in prune)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.post(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
