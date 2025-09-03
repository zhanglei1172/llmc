import math

import torch
import torch.nn as nn
from loguru import logger
import geoopt
from geoopt.manifolds import EuclideanStiefel,Stiefel

from .hadamard_utils import HadamardTransform, matmul_hadU_cuda
from .constant import *

class RotateModule(nn.Module):
    def __init__(self, Q_init):
        super(RotateModule, self).__init__()
        # self.weight = nn.Parameter(Q_init.to(torch.float32).to(torch.device('cuda')), requires_grad=True)
        self.weight = geoopt.ManifoldParameter(Q_init.to(dtype=torch.float32,device="cuda"),manifold=Stiefel())

    def forward(self, x, transpose=False):
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x

class SmoothModule(nn.Module):
    def __init__(self, S_init):
        super(SmoothModule, self).__init__()
        self.weight = nn.Parameter(S_init.to(torch.float32).to(torch.device("cuda")))

    def forward(self, x, inverse=False):
        if inverse:
            return x / self.weight
        else:
            return  x * self.weight   


class WeightRotater:
    def __init__(self, weight_rotate_func, dev):
        self.rotate_func = weight_rotate_func
        self.dev = dev

    def rotate(self, weight, bias, Q1, Q2, transpose):

        if Q1 is not None:
            tmp_weight, tmp_bias = self.rotate_func(weight, bias, Q1.weight, transpose)

            if Q2 is not None:
                had_dim = Q2.weight.shape[-1]
                dtype = tmp_weight.dtype
                dev = tmp_weight.device
                n = 1 if len(Q2.weight.shape) == 2 else Q2.weight.shape[0]
                Q = Q2.weight.reshape(n, 1, had_dim, had_dim)
                if transpose:
                    init_shape = tmp_weight.shape
                    tmp_weight = tmp_weight.reshape(init_shape[0], n, -1, had_dim)
                    tmp_weight = torch.einsum('anrx,nrxy->anry', tmp_weight.to(dtype=torch.float64), Q.to(device=dev,dtype=torch.float64)).to(dtype=dtype, device=ROTATE_DEV)
                    tmp_weight = tmp_weight.reshape(init_shape)
                else:
                    tmp_weight = tmp_weight.t()
                    transposed_shape = tmp_weight.shape
                    tmp_weight = tmp_weight.reshape(transposed_shape[0], n, -1, had_dim)
                    tmp_weight = torch.einsum('anrx,nrxy->anry', tmp_weight.to(dtype=torch.float64), Q.to(device=dev,dtype=torch.float64)).to(dtype=dtype, device=ROTATE_DEV)
                    tmp_weight = tmp_weight.reshape(transposed_shape).t()
                if bias is not None and not transpose:
                    dtype = bias.dtype
                    dev = bias.device
                    bias_shape = bias.shape
                    tmp_bias = tmp_bias.reshape(n, -1, had_dim)
                    tmp_bias = torch.einsum('nrx,nrxy->nry', tmp_bias.to(dtype=torch.float64), Q.to(device=dev,dtype=torch.float64)).to(dtype=dtype, device=ROTATE_DEV).to(device=ROTATE_DEV, dtype=dtype).reshape(bias_shape)

        if Q1 is None and Q2 is None:
            tmp_weight = weight
            tmp_bias = bias

        tmp_weight = tmp_weight.to(self.dev)
        tmp_bias = tmp_bias.to(self.dev) if tmp_bias is not None else None

        return tmp_weight, tmp_bias

class WeightRotaterSmooth:
    def __init__(self, weight_rotate_func, dev):
        self.rotate_func = weight_rotate_func
        self.dev = dev

    def rotate(self, weight, bias, Q1, Q2, transpose, Sin, Sout, inverse_out=False, head_dim=None, is_qk=False):

        if Q1 is not None:
            tmp_weight, tmp_bias = self.rotate_func(weight, bias, Q1.weight, transpose)

            if Q2 is not None:
                had_dim = Q2.weight.shape[-1]
                dtype = tmp_weight.dtype
                dev = tmp_weight.device
                n = 1 if len(Q2.weight.shape) == 2 else Q2.weight.shape[0]
                Q = Q2.weight.reshape(n, 1, had_dim, had_dim)
                if transpose:
                    init_shape = tmp_weight.shape
                    tmp_weight = tmp_weight.reshape(init_shape[0], n, -1, had_dim)
                    tmp_weight = torch.einsum('anrx,nrxy->anry', tmp_weight.to(dtype=torch.float64), Q.to(device=dev,dtype=torch.float64)).to(dtype=dtype, device=ROTATE_DEV)
                    tmp_weight = tmp_weight.reshape(init_shape)
                else:
                    tmp_weight = tmp_weight.t()
                    transposed_shape = tmp_weight.shape
                    tmp_weight = tmp_weight.reshape(transposed_shape[0], n, -1, had_dim)
                    tmp_weight = torch.einsum('anrx,nrxy->anry', tmp_weight.to(dtype=torch.float64), Q.to(device=dev,dtype=torch.float64)).to(dtype=dtype, device=ROTATE_DEV)
                    tmp_weight = tmp_weight.reshape(transposed_shape).t()
                if bias is not None and not transpose:
                    dtype = bias.dtype
                    dev = bias.device
                    bias_shape = bias.shape
                    tmp_bias = tmp_bias.reshape(n, -1, had_dim)
                    tmp_bias = torch.einsum('nrx,nrxy->nry', tmp_bias.to(dtype=torch.float64), Q.to(device=dev,dtype=torch.float64)).to(dtype=dtype, device=ROTATE_DEV).to(device=ROTATE_DEV, dtype=dtype).reshape(bias_shape)

        if Q1 is None and Q2 is None:
            tmp_weight = weight
            tmp_bias = bias

        if Sin is not None:
            dtype = tmp_weight.dtype
            dev = tmp_weight.device
            S = Sin.weight
            S = S.to(device=dev, dtype=torch.float64)
            if tmp_weight.shape[1] != S.numel():
                S = S.view(1, -1, head_dim)
                S = torch.repeat_interleave(S, dim=1, repeats=tmp_weight.shape[1]//S.numel())
            tmp_weight = (tmp_weight.to(device=dev, dtype=torch.float64) / S.view(1, -1)).to(device=ROTATE_DEV, dtype=dtype)
        if Sout is not None:
            dtype = tmp_weight.dtype
            dev = tmp_weight.device
            S = Sout.weight
            S = S.to(device=dev, dtype=torch.float64)
            if is_qk:
                S = S.reshape(-1, head_dim//2).repeat([1, 2]).reshape(-1)
            if tmp_weight.shape[0] != S.numel():
                S = S.view(1, -1, head_dim)
                S = torch.repeat_interleave(S, dim=1, repeats=tmp_weight.shape[0]//S.numel()).flatten()                    
            if inverse_out:
                tmp_weight = (tmp_weight.to(device=dev, dtype=torch.float64) / S.view(-1, 1)).to(device=ROTATE_DEV, dtype=dtype)
            else:
                if tmp_weight.dim() == 1:
                    # tmp_weight = (tmp_weight.to(S.dtype) * S).to(device=ROTATE_DEV, dtype=dtype)
                    tmp_weight = (tmp_weight.to(device=dev, dtype=torch.float64) * S).to(device=ROTATE_DEV, dtype=dtype)
                else:
                    tmp_weight = (tmp_weight.to(device=dev, dtype=torch.float64) * S.view(-1, 1)).to(device=ROTATE_DEV, dtype=dtype)
            if tmp_bias is not None:
                if inverse_out:
                    tmp_bias = (tmp_bias.to(device=dev, dtype=torch.float64) / S).to(device=ROTATE_DEV, dtype=dtype)
                else:
                    tmp_bias = (tmp_bias.to(device=dev, dtype=torch.float64) * S).to(device=ROTATE_DEV, dtype=dtype)

        tmp_weight = tmp_weight.to(self.dev)
        tmp_bias = tmp_bias.to(self.dev) if tmp_bias is not None else None

        return tmp_weight, tmp_bias

class ActRotater:
    def __init__(
        self, online_full_had, online_partial_had, fp32_had, K, had_K=None, had_dim=None
    ):
        self.online_full_had = online_full_had
        self.online_partial_had = online_partial_had
        self.fp32_had = fp32_had
        self.K = K
        self.had_K = had_K
        self.had_dim = had_dim

    def rotate(self, x):
        x_dtype = x.dtype

        if self.online_full_had:
            if self.fp32_had:
                x = matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else:
                x = matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            if self.fp32_had:
                x = x.float()
            init_shape = x.shape
            if self.K == 1:
                x = (
                    HadamardTransform.apply(
                        x.reshape(
                            -1, init_shape[-1] // self.had_dim, self.had_dim
                        ).transpose(1, 2)
                    )
                    / math.sqrt(init_shape[-1] // self.had_dim)
                ).transpose(1, 2)
            else:
                self.had_K = self.had_K.to(x.device)

                x = (
                    self.had_K.to(x.dtype)
                    @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
                ) / math.sqrt(init_shape[-1] // self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        return x