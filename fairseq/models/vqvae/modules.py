import math
import torch
import numpy as np
from collections import OrderedDict

from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Jitter(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        prob = torch.Tensor([p / 2, 1 - p, p / 2])
        self.register_buffer("prob", prob)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        else:
            batch_size, channels, sample_size = x.size()
            feature = x.clone()

            dist = Categorical(self.prob)
            index = dist.sample(torch.Size([batch_size, sample_size])) - 1
            index[:, 0].clamp_(0, 1)
            index[:, -1].clamp_(-1, 0)
            index += torch.arange(sample_size, device=x.device)

            x = torch.gather(feature, 2, index.unsqueeze(1).expand(-1, channels, -1))
        return x
        

def make_conv(n_in, n_out, k, stride=1, padding=None, bias=True, transpose=False, weight_norm=False, init_method="kaiming"):
    if padding is None:
        padding = (k-1)//2
    if not transpose:
        conv = nn.Conv1d(n_in, n_out, k, stride, padding, bias=bias)
    else:
        conv = nn.ConvTranspose1d(n_in, n_out, k, stride, padding, bias=bias)
    if weight_norm:
        conv = nn.utils.weight_norm(conv)
    if init_method == "kaiming":
        nn.init.kaiming_normal(conv.weight)
    return conv


class ResWrapper(nn.Module):
    def __init__(self, layer, weight=1.0):
        super(ResWrapper, self).__init__()
        self.layer = layer
        self.weight = weight

    def forward(self, x):
        def module_fwd(x, layer):
            if isinstance(layer, nn.ModuleList) or isinstance(layer, nn.Sequential):
                for cl in layer:
                    x = module_fwd(x, cl)
            else:
                x = layer(x)
            return x
        res = x
        x = module_fwd(x, self.layer)
        return res + self.weight * x


class Residual(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_layer, post_gain=1.0, weight_norm=True, init_method="none"):
        super(Residual, self).__init__()

        self.id_path = make_conv(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.post_gain = post_gain

        layers = []
        in_channel = n_in
        for i in range(n_layer):
            if i != n_layer - 1:
                layers.append(
                    (f"relu_{i}", nn.ReLU(True))
                )
                layers.append(
                    (f"conv_{i}", make_conv(in_channel, n_hid, 3, weight_norm=weight_norm, init_method=init_method, bias=False))
                )
                in_channel = n_hid
            else:
                layers.append(
                    (f"relu_{i}", nn.ReLU(True))
                )
                layers.append(
                    (f"conv_{i}", make_conv(in_channel, n_out, 1, weight_norm=weight_norm, init_method=init_method, bias=False))
                )
        self.res_path = nn.Sequential(OrderedDict(layers))
    
    def forward(self, x):
        return self.id_path(x) + self.post_gain * self.res_path(x)

class ResidualStack(nn.Module):

    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        
        self._num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList(
            [Residual(in_channels, num_hiddens, num_residual_hiddens)] * self._num_residual_layers)
        self.relu = nn.ReLU()

        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = x + self.layer(x)
        return self.relu(x)
