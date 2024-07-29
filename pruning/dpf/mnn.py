import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


def liconv_bits(n):
    if isinstance(m, MaskConv2d):
        m.n_bits = n
    if isinstance(m, MaskLinear):
        m.n_bits = n

    

class Masker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, n_bits, interval):
        return x * mask
 
    @staticmethod
    def backward(ctx, grad):
        return grad, None
    
class None_LSQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
 
    @staticmethod
    def backward(ctx, grad):
        return grad
    

class Full_part(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, x, mask, n_bits, interval):

        prunned_weight = x 

        lower = -2 ** (n_bits - 1) + 1
        upper = 2 ** (n_bits - 1)

        constraint = np.arange(lower, upper)
        ctx.valmin = float(constraint.min())
        ctx.valmax = float(constraint.max())

        ##################################################

        x_min = min(0., float(prunned_weight.min()))
        x_max = max(0., float(prunned_weight.max()))
        x_scale = torch.div(prunned_weight, interval)
        x_clip = F.hardtanh(x_scale, min_val=ctx.valmin, max_val=ctx.valmax)
        x_round = torch.round(x_clip)
        x_restore = torch.mul(x_round, interval)


        ctx.save_for_backward(mask,x_clip,interval)
        return x_restore

    @staticmethod
    def backward(ctx, grad):
        mask,x_clip,interval = ctx.saved_tensors

        internal_flag = ((x_clip > ctx.valmin) * (x_clip < ctx.valmax)).float()
        

        # gradient for scale
        grad_one = x_clip * internal_flag
        grad_two = torch.round(x_clip)
        grad_scale_elem = grad_two - grad_one
        grad_interval = (grad_scale_elem * grad).sum().view((1,))

        return grad, None, None, grad_interval


class Masker_part(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, x, mask, n_bits, interval):

        prunned_weight = x * mask

        lower = -2 ** (n_bits - 1) + 1
        upper = 2 ** (n_bits - 1)

        constraint = np.arange(lower, upper)
        ctx.valmin = float(constraint.min())
        ctx.valmax = float(constraint.max())

        ##################################################

        x_min = min(0., float(prunned_weight.min()))
        x_max = max(0., float(prunned_weight.max()))
        x_scale = torch.div(prunned_weight, interval)
        x_clip = F.hardtanh(x_scale, min_val=ctx.valmin, max_val=ctx.valmax)
        x_round = torch.round(x_clip)
        x_restore = torch.mul(x_round, interval)
        
        ctx.save_for_backward(mask,x_clip,interval)
        return x_restore

    @staticmethod
    def backward(ctx, grad):
        mask,x_clip,interval = ctx.saved_tensors

        internal_flag = ((x_clip > ctx.valmin) * (x_clip < ctx.valmax)).float()
        

        # gradient for scale
        grad_one = x_clip * internal_flag
        grad_two = torch.round(x_clip)
        grad_scale_elem = grad_two - grad_one
        grad_interval = (grad_scale_elem * grad).sum().view((1,))

        return grad, None, None, grad_interval


class Masker_part_fixed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, n_bits, interval):
        
        prunned_weight = x*mask

        lower = -2 ** (n_bits - 1) + 1
        upper = 2 ** (n_bits - 1)

        constraint = np.arange(lower, upper)
        ctx.valmin = float(constraint.min())
        ctx.valmax = float(constraint.max())

        ##################################################

        x_min = min(0., float(prunned_weight.min()))
        x_max = max(0., float(prunned_weight.max()))
        x_scale = torch.div(prunned_weight, interval)
        x_clip = F.hardtanh(x_scale, min_val=ctx.valmin, max_val=ctx.valmax)
        x_round = torch.round(x_clip)
        x_restore = torch.mul(x_round, interval)
                
        ctx.save_for_backward(mask,x_clip,interval)
        return x_restore

    @staticmethod
    def backward(ctx, grad):
        mask,x_clip,interval = ctx.saved_tensors

        internal_flag = ((x_clip > ctx.valmin) * (x_clip < ctx.valmax)).float()
        
        # gradient for scale
        grad_one = x_clip * internal_flag
        grad_two = torch.round(x_clip)
        grad_scale_elem = grad_two - grad_one
        grad_interval = (grad_scale_elem * grad).sum().view((1,))

        return grad * mask, None, None, grad_interval


class Masker_full(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, n_bits, interval):
        ctx.save_for_backward(mask)
        return x

    @staticmethod
    def backward(ctx, grad):
        mask, = ctx.saved_tensors
        return grad*(1-mask), None
    

class MaskConv2d(nn.Conv2d):
    def __init__(self,n_bit, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',scale_it=None):
        super(MaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias, padding_mode)
        self.mask = nn.Parameter(torch.ones(self.weight.size()), requires_grad=False)
        
        scale_init = (self.weight.abs().mean() * 2 / ((2 ** n_bit -1) ** 0.5)).cuda()
        self.interval = nn.Parameter(scale_init)
        
        # 0 -> part use, 1-> full use
        self.type_value = 0
        self.n_bits = 0
        self.acti_quan = 0


    def forward(self, input):
        # print(self.interval) 
        if self.type_value == 0:
            masked_weight = Masker_part.apply(self.weight, self.mask, self.n_bits, self.interval)

        elif self.type_value == 1:
            masked_weight = Masker_part_fixed.apply(self.weight, self.mask, self.n_bits, self.interval)

        elif self.type_value == 2:
            masked_weight = Full_part.apply(self.weight, self.mask, self.n_bits, self.interval)
        # not LSQ
        elif self.type_value == 3:
            masked_weight = None_LSQ.apply(self.weight)            

        return super(MaskConv2d, self)._conv_forward(input, masked_weight,self.bias)

    
class MaskLinear(nn.Linear):
    def __init__(self,n_bit, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__(in_features, out_features, bias)
        self.mask = nn.Parameter(torch.ones(self.weight.size()), requires_grad=False)
        self.type_value = 0
        self.n_bits = 0
        self.acti_quan = 0
        scale_init = (self.weight.abs().mean() * 2 / ((2 ** n_bit -1) ** 0.5)).cuda()
        # #       scale_init = torch.ones(1)
        self.interval = nn.Parameter(scale_init)
        

    def forward(self, input):
        if self.type_value == 0:
            masked_weight = Masker_part.apply(self.weight, self.mask, self.n_bits, self.interval)

        elif self.type_value == 1:
            masked_weight = Masker_part_fixed.apply(self.weight, self.mask, self.n_bits, self.interval)
            
        elif self.type_value == 2:
            masked_weight = Full_part.apply(self.weight, self.mask, self.n_bits, self.interval)
        # not LSQ
        elif self.type_value == 3:
            masked_weight = None_LSQ.apply(self.weight)
            
        elif self.type_value == 4:
            masked_weight = Masker.apply(self.weight, self.mask, self.n_bits, self.interval)

            

        return F.linear(input, masked_weight,self.bias)
        
    
class LsqActivationFun(torch.autograd.Function):
    def forward(ctx, x, n_bits, interval):
        lower = 0
        upper = 2 ** n_bits - 1

        constraint = np.arange(lower, upper)
        ctx.valmin = float(constraint.min())
        ctx.valmax = float(constraint.max())
        

        ##################################################

        x_min = min(0., float(x.min()))
        x_max = max(0., float(x.max()))

        x_scale = torch.div(x, interval)
        x_clip = F.hardtanh(x_scale, min_val=ctx.valmin, max_val=ctx.valmax)
        x_round = torch.round(x_clip)
        x_restore = torch.mul(x_round, interval)
        
        
        ctx.save_for_backward(x_clip,interval)
        return x_restore

    @staticmethod
    def backward(ctx, grad):
        x_clip,interval = ctx.saved_tensors
        internal_flag = ((x_clip > ctx.valmin) * (x_clip < ctx.valmax)).float()
        
        grad_activation = grad * internal_flag
        
        # gradient for scale
        grad_one = x_clip * internal_flag
        grad_two = torch.round(x_clip)
        grad_scale_elem = grad_two - grad_one
        grad_interval = (grad_scale_elem * grad).sum().view((1,))

        return grad_activation , None, grad_interval
    
class LsqActivation(nn.Module):
    def __init__(self,x,acti_n_bits,scale_it=None):
        super(LsqActivation, self).__init__()
        scale_it = (x.detach().abs().mean() * 2 / ((2 ** acti_n_bits -1) ** 0.5)).cuda()
        self.scale = nn.Parameter(scale_it)
        self.acti_n_bits = acti_n_bits

    def forward(self, x):
        return LsqActivationFun.apply(x, self.acti_n_bits, self.scale)
