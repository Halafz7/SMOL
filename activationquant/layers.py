import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from ops import *

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class PACT(nn.Module):
  def __init__(self, shape, p, alpha):
    super(PACT, self).__init__()
    self.alpha = nn.Parameter(torch.tensor(alpha))
    self.s = None
    self.p = torch.tensor(p)

  def forward(self, x):
    x = torch.clip(x, max=6)

    clipped = ClipFunc.apply(x, self.alpha)
    normalized = clipped / self.alpha.detach()
    quantized = Quant.apply(normalized, self.p)
    scaled = quantized * self.alpha.detach()
    return scaled

class REPACT(nn.Module):
  def __init__(self, shape, p, alpha):
    super(REPACT, self).__init__()

    alpha = 1/alpha
    ainit = math.log(alpha / (1 - alpha))
    self.alpha = nn.Parameter(torch.tensor(ainit))
    self.s = None
    self.p = torch.tensor(p)

    self.register_buffer('ema', torch.tensor(0.0))
    self.register_buffer('t', torch.tensor(0))

  def forward(self, x):
    #x = torch.clip(x, max=6)

    if self.training:
        self.t += 1
        self.ema = 0.9 * self.ema.detach() + 0.1 * torch.max(x)
    ema_corrected = self.ema / (1 - 0.9**self.t)

    beta = torch.sigmoid(self.alpha) * ema_corrected

    clipped = ClipFunc.apply(x, beta)
    normalized = clipped / beta.detach()
    quantized = Quant.apply(normalized, self.p)
    scaled = quantized * beta.detach()
    return scaled

class PELT(nn.Module):
    def __init__(self, shape, p_init, alpha):
        super(PELT, self).__init__()
        self.mode = 'noisy'
        self.qf = torch.floor

        s = math.log(2**(1-p_init) / (1 - 2**(1-p_init)))
        self.s = nn.Parameter(torch.ones(shape)*s)
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.p = None

    def forward(self, x):
        x = torch.clip(x, max=6)

        clipped = ClipFunc.apply(x, self.alpha)
        if self.mode == 'quant':
            self.p = self.qf(1-torch.log2(torch.sigmoid(self.s)))
            u = Quant.apply(clipped/self.alpha.detach(), self.p) * self.alpha.detach()
        elif self.mode == 'noisy':
            u = clipped + self.alpha * torch.sigmoid(self.s)/2 * torch.empty_like(self.s).uniform_(-1, 1) 
        return u

class PELT2(nn.Module):
    # PELT but with scale reparameterization, such that we use beta = alpha * max(x)!
    def __init__(self, shape, p_init, alpha):
        super(PELT2, self).__init__()
        self.mode = 'noisy'
        self.qf = torch.floor

        s = math.log(2**(1-p_init) / (1 - 2**(1-p_init)))
        self.s = nn.Parameter(torch.ones(shape)*s)
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.p = None

    def forward(self, x):
        scale = x.max()
        beta = self.alpha*scale
        clipped = ClipFunc.apply(x, beta)
        if self.mode == 'quant':
            self.p = self.qf(1-torch.log2(torch.sigmoid(self.s)))
            u = Quant.apply(clipped/beta.detach(), self.p) * beta.detach()
        elif self.mode == 'noisy':
            u = clipped + beta.detach() * torch.sigmoid(self.s)/2 * torch.empty_like(self.s).uniform_(-1, 1)
        return u

class ClipFunc(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return torch.clamp(x, min = 0, max = alpha.item())

    @staticmethod
    def backward(ctx, dLdy_q):
        x, alpha, = ctx.saved_tensors
        lower_bound = x < 0
        upper_bound = x > alpha
        x_range = ~(lower_bound|upper_bound)
        grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1)
        return dLdy_q * x_range.float(), grad_alpha

class ClipFuncPtws(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return torch.minimum(torch.clamp(x, min = 0), alpha)

    @staticmethod
    def backward(ctx, dLdy_q):
        x, alpha, = ctx.saved_tensors
        lower_bound = x < 0
        upper_bound = x > alpha
        x_range = ~(lower_bound|upper_bound)
        grad_alpha = dLdy_q * torch.ge(x, alpha).float()
        return dLdy_q * x_range.float(), grad_alpha

class Quant(Function):
    @staticmethod
    def forward(ctx, x, p):
        scale = (2**p - 1)
        y = torch.clamp(x, min = 0, max = 1)
        y_q = torch.round(y * scale) / scale
        return y_q

    @staticmethod
    def backward(ctx, dLdy_q):
        return dLdy_q, None

qamodules = [PACT, REPACT, PELT, PELT2]