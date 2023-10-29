import torch
from torch.autograd import Function

class Quantize(Function):
    @staticmethod
    def forward(ctx, w, s, qf):
        return quantize(w, s, qf)
        
    @staticmethod
    def backward(ctx, gradOutput, _):
        return gradOutput, gradOutput*0, None

def quantize(w, s, qf):
    mag_w = torch.max(torch.abs(w)).item()
    norm2, norm2_p = general_quantize(w, s, qf, 0, mag_w/2, prune=False)
    return norm2, norm2_p


def general_quantize(w, s, qf, offset, delta, prune=False, norm=True):
    if norm:
        scaling = max(1e-6, min(1, torch.abs(w).max().item()))
        p = qf(1-torch.log2(torch.sigmoid(s)/scaling))
        p = torch.clamp(p, min=1)
    else:
        p = qf(1-torch.log2(torch.sigmoid(s)))

    alpha = 2**(p-2) / delta
    beta = delta*(2-2**(1-p)) + offset

    w_scaled = alpha * (w + beta)
    w_rounded = torch.round(w_scaled)
    w_rounded = torch.clamp(w_rounded, min=0)
    w_rounded = torch.minimum(w_rounded, 2**p-1)
    w_quant = w_rounded/alpha - beta

    if prune:
        quant_err = torch.abs(w - w_quant)
        prune_err = torch.abs(w)
        keep_bool = prune_err > quant_err
        p = p * keep_bool
        w_quant = w_quant * keep_bool

    return w_quant, p