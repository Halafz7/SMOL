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
            cost += -torch.log2(torch.sigmoid(m.weight_s)).sum()
            if m.bias is not None: cost += -torch.log2(torch.sigmoid(m.bias_s)).sum()
        return cost

    def project(self):
        for m in self.mixed_layers:
            p = 1-torch.log2(torch.sigmoid(m.weight_s))
            M = 2-2**(1-p)

            if torch.__version__ == '1.1.0':
                maxf, minf = torch.max, torch.min
            else:
                maxf, minf = torch.maximum, torch.minimum

            m.weight.data = minf(m.weight.data, M)
            m.weight.data = maxf(m.weight.data, -M)

            if m.bias is not None:
                p = 1-torch.log2(torch.sigmoid(m.bias_s))
                M = 2-2**(1-p)
                m.bias.data = minf(m.bias.data, M)
                m.bias.data = maxf(m.bias.data, -M)

    def print_precs(self):
        precs = 0
        norm_precs = 0
        num_precs = 0
        for m in self.mixed_layers:
            if m.p is not None:
                precs += m.p.sum()
                norm_precs = precs
                num_precs += m.p.numel()

            else:
                scaling = max(1e-6, min(1, torch.abs(m.weight).max().item()))
                noise_mag = torch.sigmoid(m.weight_s)

                p_float = 1-torch.log2(noise_mag)
                p_float = torch.clamp(p_float, min=1)

                norm_p_float = 1-torch.log2(noise_mag/scaling)
                norm_p_float = torch.clamp(norm_p_float, min=1)

                precs += self.qf(p_float).sum()
                norm_precs += self.qf(norm_p_float).sum()
                num_precs += m.weight_s.numel()

            if m.bias is not None:
                if m.bp is not None:
                    precs += m.bp.sum()
                    norm_precs = precs
                    num_precs += m.bp.numel()
                else:
                    scaling = max(1e-6, min(1, torch.abs(m.bias).max().item()))
                    noise_mag = torch.sigmoid(m.bias_s)

                    p_float = 1-torch.log2(noise_mag)
                    p_float = torch.clamp(p_float, min=1)

                    norm_p_float = 1-torch.log2(noise_mag/scaling)
                    norm_p_float = torch.clamp(norm_p_float, min=1)

                    precs += self.qf(p_float).sum()
                    norm_precs += self.qf(norm_p_float).sum()
                    num_precs += m.bias_s.numel()

        if num_precs == 0: return "Network is in full precision"
        return "Network precision with {}: {} / {}".format(self.qf.__name__, (precs/num_precs).item(), (norm_precs/num_precs).item())

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
    def __init__(self, inplanes, planes, stride, shortcut, p_init, fp, first=False):
        super(PreActBlock, self).__init__()
        self.first = first
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv(inplanes, planes, 3, stride, p_init, fp)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv(planes, planes, 3, 1, p_init, fp)
        self.relu2 = nn.ReLU(inplace=True)
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        out = self.relu1(self.bn1(x))
        if self.first: residual = out
        out = self.relu2(self.bn2(self.conv1(out)))
        out = self.conv2(out)
        if self.shortcut is not None: residual = self.shortcut(residual)
        out += residual
        return out

class PreResNet(MixedNet):
    def __init__(self, mode='quant', qf=torch.floor, p_init=32, fp_layers='shortcuts', shortcut_type='CB'):
        super(PreResNet, self).__init__(mode, qf, p_init)
        self.p_init = p_init
        self.fp_layers = fp_layers
        self.shortcut_module = {'P':ShortcutP, 'C':ShortcutC, 'CB':ShortcutCB}[shortcut_type]

        fp_firstconv = (fp_layers=='ends' or fp_layers=='all')
        self.pre = conv(3, 16, 3, 1, p_init, fp_firstconv)
        self.stage1 = self.make_stage(16, 16, 3, 1)
        self.stage2 = self.make_stage(16, 32, 3, 2)
        self.stage3 = self.make_stage(32, 64, 3, 2)
        self.post = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.AvgPool2d(8))

        fp_fc = (fp_layers == 'all' or fp_layers == 'ends')
        self.fc = linear(64, 10, p_init, fp_fc)

        self.mixed_layers = [m for m in self.modules() if isinstance(m, NoisyConv2d) or isinstance(m, NoisyLinear)]
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_stage(self, inplanes, planes, blocks, stride):
        fp_shortcut = (self.fp_layers=='all' or self.fp_layers == 'shortcuts')
        fp_blocks = (self.fp_layers=='all')

        if stride != 1 or inplanes != planes: shortcut = self.shortcut_module(inplanes, planes, self.p_init, fp_shortcut)
        else: shortcut = None

        layers = [PreActBlock(inplanes, planes, stride, shortcut, self.p_init, fp_blocks)]
        for i in range(1, blocks): layers.append(PreActBlock(planes, planes, 1, None, self.p_init, fp_blocks))
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





class MNv2Block(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride, p_init, fp_layers):
        super(MNv2Block, self).__init__()
        self.stride = stride
        planes = expansion * in_planes

        if fp_layers == 'all':
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.conv1 = NoisyConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False, p_init=p_init)
        self.bn1 = nn.BatchNorm2d(planes)

        if fp_layers == 'all':
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=planes)
        else:
            self.conv2 = NoisyConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=planes, p_init=p_init)
        self.bn2 = nn.BatchNorm2d(planes)
        if fp_layers == 'all':
            self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.conv3 = NoisyConv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, p_init=p_init)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(MixedNet):
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, mode='quant', qf=torch.floor, p_init=32, fp_layers='shortcuts', shortcut_type='CB'):
        super(MobileNetV2, self).__init__(mode, qf, p_init)
        self.p_init = p_init
        self.fp_layers = fp_layers

        if self.fp_layers == 'all':
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = NoisyConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False, p_init=p_init)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        if self.fp_layers == 'all':
            self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.conv2 = NoisyConv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False, p_init=p_init)
        self.bn2 = nn.BatchNorm2d(1280)
        if self.fp_layers == 'all':
            self.linear = nn.Linear(1280, 10)
        else:
            self.linear = NoisyLinear(1280, 10, p_init=p_init)
        self.mixed_layers = [m for m in self.modules() if isinstance(m, NoisyConv2d) or isinstance(m, NoisyLinear)]

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(MNv2Block(in_planes, out_planes, expansion, stride, self.p_init, self.fp_layers))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out












class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)


class ShuffleBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups, p_init=32, fp_layers='none'):
        super(ShuffleBottleneck, self).__init__()
        self.stride = stride

        mid_planes = out_planes//4
        g = 1 if in_planes==24 else groups

        if fp_layers == 'all':
            self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, padding=0, bias=False, groups=g)
        else:
            self.conv1 = NoisyConv2d(in_planes, mid_planes, kernel_size=1, stride=1, padding=0, bias=False, groups=g, p_init=p_init)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)

        if fp_layers == 'all':
            self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=mid_planes)
        else:
            self.conv2 = NoisyConv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=mid_planes, p_init=p_init)
        self.bn2 = nn.BatchNorm2d(mid_planes)

        if fp_layers == 'all':
            self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, groups=groups)
        else:
            self.conv3 = NoisyConv2d(mid_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, groups=groups, p_init=p_init)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out,res], 1)) if self.stride==2 else F.relu(out+res)
        return out

class ShuffleNet(MixedNet):
    def __init__(self, mode='quant', qf=torch.floor, p_init=32, fp_layers='shortcuts', shortcut_type='CB'):
        cfg = {
            'out_planes': [200,400,800],
            'num_blocks': [4,8,4],
            'groups': 2
        }

        super(ShuffleNet, self).__init__(mode, qf, p_init)
        self.p_init = p_init
        self.fp_layers = fp_layers

        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        if fp_layers == 'all':
            self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        else:
            self.conv1 = NoisyConv2d(3, 24, kernel_size=1, padding=0, stride=1, bias=False, p_init=p_init)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        if fp_layers == 'all':
            self.linear = nn.Linear(out_planes[2], 10)
        else:
            self.linear = NoisyLinear(out_planes[2], 10, p_init=p_init)

        self.mixed_layers = [m for m in self.modules() if isinstance(m, NoisyConv2d) or isinstance(m, NoisyLinear)]

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(ShuffleBottleneck(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups, p_init=self.p_init, fp_layers=self.fp_layers))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out