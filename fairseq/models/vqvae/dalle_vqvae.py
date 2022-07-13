# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Tuple
from functools import partial
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model


def make_conv(n_in, n_out, k):
    conv = nn.Conv1d(n_in, n_out, k, padding=(k-1)//2)
    nn.init.normal_(conv.weight, std=1 / math.sqrt(n_in * k))
    return conv

@dataclass
class DalleVqvaeConfig(FairseqDataclass):
    # encoder config
    feature_dim: int = field(
        default=39, metadata={"help": "input feautre dimension, 39 for MFCC, 80 for Fbank"}
    )
    output_dim: int = field(
        default=39, metadata={"help": "output feautre dimension, 39 for MFCC, 80 for Fbank"}
    )
    encoder_conv_layers: str = field(
        default="(4, 2, 256)", 
        metadata={"help": "convolutional encoder architecture (output channel, kernel size, stride, padding)"}
    )
   
    # vq config
    vq_type: str = field(
        default="default", metadata={"help": "which type of quantizer to use"}
    )
    latent_vars: int = field(
        default=320,
        metadata={"help": "number of latent variables V in each group of the codebook"},
    )
    latent_groups: int = field(
        default=1,
        metadata={"help": "number of groups G of latent variables in the codebook"},
    )
    latent_dim: int = field(
        default=64,
        metadata={
            "help": "if > 0, uses this dimensionality for latent variables. "
            "otherwise uses final_dim / latent_groups"
        },
    )
    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={
            "help": "temperature for latent variable sampling. "
            "can be tuple of 3 values (start, end, decay)"
        },
    )
    vq_gamma: float = field(
        default=0.25,
        metadata={"help": "gamma parameter for kmeans style vector quantization"},
    )

    # decoder config

    decoder_conv_layers: str = field(
        default="(4, 2, 256)", 
        metadata={"help": "convolutional decoder architecture (output channel, kernel size, stride, padding)"}
    )
    use_jitter: bool = field(
        default=False, metadata={"help": "whether use jitter layer"}
    )
    jitter_prob: float = field(
        default=0.12, metadata={"help": "jitter probability"}
    )
    speaker_cond: bool = field(
        default=False, metadata={"help": "whether use speaker id"}
    )

class ConvBlock(nn.Module):
    def __init__(self, n_in, n_out, n_layers):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_layers = n_layers
        self.n_hid = self.n_out // 4
        self.post_gain = 1 / (self.n_layers ** 2)
        
        self.id_path  = make_conv(self.n_in, self.n_out, 1) if self.n_in != self.n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([
				('relu_1', nn.ReLU()),
				('conv_1', make_conv(self.n_in,  self.n_hid, 3)),
				('relu_2', nn.ReLU()),
				('conv_2', make_conv(self.n_hid, self.n_hid, 3)),
				('relu_3', nn.ReLU()),
				('conv_3', make_conv(self.n_hid, self.n_hid, 3)),
				('relu_4', nn.ReLU()),
				('conv_4', make_conv(self.n_hid, self.n_out, 1)),]))
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)

class Encoder(nn.Module):
    def __init__(self, n_in, group_count=4, n_hid=256, n_blk_per_group=2, var_dim=200):
        super().__init__()
        self.group_count = group_count
        self.input_channels = n_in
        self.n_hid = n_hid
        self.n_blk_per_group = n_blk_per_group
        self.var_dim = var_dim

        blk_range  = range(self.n_blk_per_group)
        n_layers   = self.group_count * self.n_blk_per_group
        make_blk   = partial(ConvBlock, n_layers=n_layers)
        
        self.blocks = nn.Sequential(OrderedDict([
			('input', make_conv(n_in, 1 * self.n_hid, 7)),
			('group_1', nn.Sequential(OrderedDict([
				*[(f'block_{i + 1}', make_blk(1 * self.n_hid, 1 * self.n_hid)) for i in blk_range],
				('pool', nn.MaxPool1d(kernel_size=2)),
			]))),
			('group_2', nn.Sequential(OrderedDict([
				*[(f'block_{i + 1}', make_blk(1 * self.n_hid if i == 0 else 2 * self.n_hid, 2 * self.n_hid)) for i in blk_range],
				('pool', nn.MaxPool1d(kernel_size=2)),
			]))),
			('group_3', nn.Sequential(OrderedDict([
				*[(f'block_{i + 1}', make_blk(2 * self.n_hid if i == 0 else 4 * self.n_hid, 4 * self.n_hid)) for i in blk_range],
				('pool', nn.MaxPool1d(kernel_size=2)),
			]))),
			('group_4', nn.Sequential(OrderedDict([
				*[(f'block_{i + 1}', make_blk(4 * self.n_hid if i == 0 else 8 * self.n_hid, 8 * self.n_hid)) for i in blk_range],
			]))),
			('output', nn.Sequential(OrderedDict([
				('relu', nn.ReLU()),
				('conv', make_conv(8 * self.n_hid, self.var_dim, 1)),
			]))),
        ]))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.input_channels:
            raise ValueError(f'input has {x.shape[1]} channels but model built for {self.input_channels}')
        
        return self.blocks(x)

class Jitter(nn.Module):
    def __init__(self, probability=0.12):
        super(Jitter, self).__init__()

        self._probability = probability

    def forward(self, x):
        feature = x.detach().clone()

        length = feature.size(2)
        for i in range(length):
            replace = [True, False][np.random.choice([1, 0], p=[self._probability, 1 - self._probability])]
            if replace:
                if i == 0:
                    neighbor_index = i + 1
                elif i == length - 1:
                    neighbor_index = i - 1
                else:
                    neighbor_index = i + np.random.choice([-1, 1], p=[0.5, 0.5])

                x[:, :, i] = feature[:, :, neighbor_index]
        return x                

class Decoder(nn.Module):
    def __init__(
        self,
        n_init: int,
        n_hid: int,
        n_output: int,
        group_count = 4,
        n_blk_per_group=2,
        use_jitter: bool = True,
        jitter_prob: float = 0.25,
        speaker_cond: bool = False
    ):
        super(Decoder, self).__init__()
        self._use_jitter = use_jitter
        self._speaker_cond = speaker_cond

        if self._use_jitter:
            self.jitter = Jitter(jitter_prob)

        input_dim = n_init + 40 if self._speaker_cond else n_init
        
        blk_range  = range(n_blk_per_group)
        n_layers   = group_count * n_blk_per_group
        make_blk   = partial(ConvBlock, n_layers=n_layers)
        
        self.blocks = nn.Sequential(OrderedDict([
			('input', make_conv(input_dim, n_hid, 1) if input_dim != n_hid else nn.Identity()),
			('group_1', nn.Sequential(OrderedDict([
				*[(f'block_{i + 1}', make_blk(n_hid if i == 0 else 8 * n_hid, 8 * n_hid)) for i in blk_range],
				('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
			]))),
			('group_2', nn.Sequential(OrderedDict([
				*[(f'block_{i + 1}', make_blk(8 * n_hid if i == 0 else 4 * n_hid, 4 * n_hid)) for i in blk_range],
				('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
			]))),
			('group_3', nn.Sequential(OrderedDict([
				*[(f'block_{i + 1}', make_blk(4 * n_hid if i == 0 else 2 * n_hid, 2 * n_hid)) for i in blk_range],
				('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
			]))),
			('group_4', nn.Sequential(OrderedDict([
				*[(f'block_{i + 1}', make_blk(2 * n_hid if i == 0 else 1 * n_hid, 1 * n_hid)) for i in blk_range],
			]))),
			('output', nn.Sequential(OrderedDict([
				('relu', nn.ReLU()),
				('conv', make_conv(1 * n_hid, n_output, 1)),
			]))),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:            
        if self._use_jitter and self.training:
            x = self.jitter(x)

        if self._speaker_cond:
            speaker_embedding = GlobalConditioning.compute(
                speaker_dic, speaker_id, x, gin_channel=40, expand=True
            )
            x = torch.cat([x, speaker_embedding], dim=1)
            
        return self.blocks(x)
        


@register_model("dalle_vqvae", dataclass=DalleVqvaeConfig)
class DalleVqvaeModel(BaseFairseqModel):
    def __init__(self, cfg: DalleVqvaeConfig):
        super(DalleVqvaeModel, self).__init__()
        self.cfg = cfg
        (group_count, n_blk_per_group, enc_hid) = eval(cfg.encoder_conv_layers)

        self.encoder = Encoder(
            n_in=cfg.feature_dim,
            group_count=group_count,
            n_hid=enc_hid,
            n_blk_per_group=n_blk_per_group,
            var_dim=cfg.latent_vars * cfg.latent_groups
        )
        self.groups = cfg.latent_groups
        self.latent_vars = cfg.latent_vars
        self.max_temp, self.min_temp, self.temp_decay = cfg.latent_temp
        self.curr_temp = self.max_temp

        self.vars = nn.Parameter(torch.FloatTensor(1, cfg.latent_groups * cfg.latent_vars, cfg.latent_dim))
        nn.init.uniform_(self.vars)

        (group_count, n_blk_per_group, dec_hid) = eval(cfg.decoder_conv_layers)

        self.decoder = Decoder(
            n_init=cfg.latent_dim,
            n_hid=dec_hid,
            n_output=cfg.output_dim,
            group_count=group_count,
            n_blk_per_group=n_blk_per_group,
            use_jitter=cfg.use_jitter,
            jitter_prob=cfg.jitter_prob,
            speaker_cond=cfg.speaker_cond,
        )
        
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, cfg: DalleVqvaeConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)
    

    def set_num_updates(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    def forward(
        self,
        source,
        speaker_id=None,
        features_only=False
    ):
        source = source.permute(0, 2, 1).contiguous()
        x = self.encoder(source)
        features_pen = x.float().pow(2).mean()

        x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
        x = x.contiguous().view(bsz * tsz * self.groups, -1)

        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result = {"num_vars": self.latent_vars * self.groups}
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.groups, -1).float(), dim=-1
        ).mean(dim=0)
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        result["temp"] = self.curr_temp

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        vars = self.vars

        z = x.unsqueeze(-1) * vars
        z = z.view(bsz * tsz, self.groups, self.latent_vars, -1)
        z = z.sum(-2)
        z = z.view(bsz, tsz, -1)

        z = z.transpose(1, 2)  # BTC -> BCT

        y = self.decoder(z)
        y = y.permute(0, 2, 1).contiguous()

        result["features_pen"] = features_pen
        result["reconstructed_x"] = y
        return result

        
    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(self, source, padding_mask, mask=False, output_layer=None):
        res = self.forward(
            source, padding_mask, mask=mask, features_only=True, output_layer=output_layer
        )
        feature = res["quantized"]
        return feature

    
    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen
