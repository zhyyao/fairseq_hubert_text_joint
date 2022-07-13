# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Tuple
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *
from fairseq import utils
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import KmeansVectorQuantizer, GumbelVectorQuantizer
from fairseq.models.vqvae.vector_quantizer import VQEmbedding, VQEmbeddingEMA



@dataclass
class CnnVqvaeConfig(FairseqDataclass):
    # encoder config
    feature_dim: int = field(
        default=39, metadata={"help": "input feautre dimension, 39 for MFCC, 80 for Fbank"}
    )
    encoder_conv_layers: str = field(
        default="[(768, 3, 1, 1)] * 2 + [(768, 4, 2, 2)] + [(768, 3, 1, 1)] * 2", 
        metadata={"help": "convolutional encoder architecture (output channel, kernel size, stride, padding)"}
    )
    encoder_conv_act: str = field(
        default="[True] * 5",
        metadata = {"help": "whether use relu activation for encoder convolutions"}
    )
    encoder_conv_res: str = field(
        default="[False, True, False, True, True]", 
        metadata = {"help": "whether use residual connection for encoder convolutions"}
    )
    encoder_residual_layers: int = field(
        default=2
    )
    pre_vq_layers: str = field(
        default="(64, 3, 1, 1)", 
        metadata={"help": "convolutional encoder architecture (output channel, kernel size, stride, padding)"}
    )
    weight_norm: bool = field(
        default=True
    )
    conv_init: str = field(
        default="kaiming"
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
        default=2,
        metadata={"help": "number of groups G of latent variables in the codebook"},
    )
    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={
            "help": "temperature for latent variable sampling. "
            "can be tuple of 3 values (start, end, decay)"
        },
    )

    latent_dim: int = field(
        default=64,
        metadata={
            "help": "if > 0, uses this dimensionality for latent variables. "
            "otherwise uses final_dim / latent_groups"
        },
    )
    vq_gamma: float = field(
        default=0.25,
        metadata={"help": "gamma parameter for kmeans style vector quantization"},
    )

    # decoder config

    decoder_conv_layers: str = field(
        default="(768, 3, 1, 1)", 
        metadata={"help": "convolutional decoder architecture (output channel, kernel size, stride, padding)"}
    )
    decoder_residual_layers: int = field(
        default=2
    )
    decoder_conv_transpose: str = field(
        default="[(512, 3, 1, 1)] + [(512, 3, 1, 1)] + [(512, 2, 1, 1)]"
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

class ConvolutionalEncoder(nn.Module):

    def __init__(
        self, 
        input_dim: int,
        conv_layers: List[Tuple[int, int, int, int]],
        do_act: List[bool],
        do_res: List[bool],
        residual_layers: int,
        weight_norm: bool=True,
        init_method: str="kaiming"
    ):

        super(ConvolutionalEncoder, self).__init__()
        assert len(conv_layers) == len(do_act), len(conv_layers) == len(do_res)
        in_d = input_dim
        layers = []
        
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 4, "invalid conv definition: " + str(cl)
            (dim, k, stride, padding) = cl
            conv = make_conv(in_d, dim, k, stride, padding, weight_norm=weight_norm, init_method=init_method)
            if do_res[i]:
                if do_act[i]:
                    layers.append((
                        f"resconv_{i+1}", ResWrapper(nn.Sequential(OrderedDict(
                            [(f"conv_{i+1}", conv), (f"relu_{i+1}", nn.ReLU())])
                    ))))
                else:
                    layers.append((f"conv_{i+1}",ResWrapper(conv)))
            else:
                layers.append((f"conv_{i+1}", conv))
        
                if do_act[i]:
                    layers.append((f"relu_{i+1}", nn.ReLU()))

            in_d = dim

        layers.append((
            "res_group", ResWrapper(nn.Sequential(OrderedDict([
                (f'block', nn.ModuleList([Residual(in_d, dim, dim, 2, weight_norm=weight_norm, init_method=init_method)] * residual_layers)),
                ('relu', nn.ReLU())]
        )))))    
            
        self.blocks = nn.Sequential(OrderedDict(layers))
      
    def forward(self, x):
        return self.blocks(x)
                

class DeconvolutionalDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        conv_layers: List[Tuple[int, int, int, int]],
        conv_transpose: List[Tuple[int, int, int, int]],
        residual_layers: int,
        use_jitter: bool = True,
        jitter_prob: float = 0.25,
        speaker_cond: bool = False,
        weight_norm=True,
        init_method="kaiming"
    ):
        super(DeconvolutionalDecoder, self).__init__()
        self._use_jitter = use_jitter
        self._speaker_cond = speaker_cond

        if self._use_jitter:
            self.jitter = Jitter(jitter_prob)

        input_dim = input_dim + 40 if self._speaker_cond else input_dim
        (dim, k, stride, padding) = conv_layers

        layers = []

        self.conv =  make_conv(input_dim, dim, k, stride, padding, weight_norm=weight_norm, init_method=init_method)

        self.upsample = nn.Upsample(scale_factor=2)
        self.res_block = nn.ModuleList([Residual(dim, dim, dim, 2, weight_norm=weight_norm, init_method=init_method)] * residual_layers)
        self.relu = nn.ReLU()
        in_d = dim


        for i, cl in enumerate(conv_transpose):
            assert len(cl) == 4, "invalid conv definition: " + str(cl)
            (dim, k, stride, padding) = cl

            layers.append(
                (f"conv_{i+2}", make_conv(in_d, dim, k, stride, padding, transpose=True, weight_norm=weight_norm, init_method=init_method))               
            )
            if i < len(conv_transpose) - 1:
                layers.append((f"relu_{i+2}", nn.ReLU()))
                
            in_d = dim
        self.blocks = nn.Sequential(OrderedDict(layers))
        
    def forward(self, x, speaker_dic=None, speaker_id=None):
        if self._use_jitter and self.training:
            x = self.jitter(x)
        
        if self._speaker_cond:
            speaker_embedding = GlobalConditioning.compute(
                speaker_dic, speaker_id, x, gin_channel=40, expand=True
            )
            x = torch.cat([x, speaker_embedding], dim=1)

        x = self.conv(x)
        x = self.upsample(x)
        for layer in self.res_block:
            x = layer(x)
        x = self.relu(x)

        x = self.blocks(x)
        return x


@register_model("cnn_vqvae", dataclass=CnnVqvaeConfig)
class CnnVqvaeModel(BaseFairseqModel):
    def __init__(self, cfg: CnnVqvaeConfig):
        super(CnnVqvaeModel, self).__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.encoder_conv_layers)

        self.encoder = ConvolutionalEncoder(
            input_dim=cfg.feature_dim,
            conv_layers=feature_enc_layers,
            do_act=eval(cfg.encoder_conv_act),
            do_res=eval(cfg.encoder_conv_res),
            residual_layers=cfg.encoder_residual_layers,
            weight_norm=cfg.weight_norm,
            init_method=cfg.conv_init
        )

        final_dim = feature_enc_layers[-1][0]
        pre_vq_layers = eval(cfg.pre_vq_layers)
        dim, k, stride, padding = pre_vq_layers
        self.pre_vq_conv = make_conv(final_dim, dim, k, stride, padding, weight_norm=cfg.weight_norm, init_method=cfg.conv_init)
            
        vq_dim = cfg.latent_dim

        if cfg.vq_type == "gumbel":
            self.vector_quantizer = GumbelVectorQuantizer(
                dim=dim,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=1,
                weight_proj_factor=3,
            )
        elif cfg.vq_type == "kmeans":
            self.vector_quantizer = KmeansVectorQuantizer(
                dim=dim,
                num_vars=cfg.latent_vars,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=False,
                gamma=cfg.vq_gamma,
            )
        elif cfg.vq_type == "vq_ema":
            self.vector_quantizer = VQEmbeddingEMA(
                n_embeddings=cfg.latent_vars,
                embedding_dim=vq_dim,
                commitment_cost=cfg.vq_gamma
            )
        else:
            self.vector_quantizer = VQEmbedding(
                n_embeddings=cfg.latent_vars,
                embedding_dim=vq_dim,
                input_dim=vq_dim,
                commitment_cost=cfg.vq_gamma
            )


        self.decoder = DeconvolutionalDecoder(
            input_dim=vq_dim,
            output_dim=cfg.feature_dim,
            conv_layers=eval(cfg.decoder_conv_layers),
            conv_transpose=eval(cfg.decoder_conv_transpose),
            residual_layers=cfg.decoder_residual_layers,
            use_jitter=cfg.use_jitter,
            jitter_prob=cfg.jitter_prob,
            speaker_cond=cfg.speaker_cond,
            weight_norm=cfg.weight_norm,
            init_method=cfg.conv_init
        )

        self.final_proj = nn.Linear(512, 39)
        
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, cfg: CnnVqvaeConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)
    

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) // stride + 1)

        conv_cfg_list = eval(self.cfg.encoder_conv_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def forward(
        self,
        source,
        target,
        speaker_dic=None,
        speaker_id=None,
        features_only=False,
        reduce=True
    ):
        source = source.permute(0, 2, 1).contiguous()
        x = self.encoder(source)
        features = self.pre_vq_conv(x)
        features_pen = features.float().pow(2).mean()
        results = self.vector_quantizer(features.transpose(1, 2))
        z = results["x"]
        y = self.decoder(z.transpose(1, 2), speaker_dic, speaker_id)
        y = y.permute(0, 2, 1).contiguous()
        y = self.final_proj(y)

        length = target.shape[1]
        if y.shape[1] > length:
            y = y[:, :length, :]

        results["y"] = y

        return results

        
    def quantize(self, source):
        source = source.permute(0, 2, 1).contiguous()
        x = self.encoder(source)
        features = self.pre_vq_conv(x)
        res = self.vector_quantizer(features.transpose(1, 2))
        feature = res["encoding_indices"]
        return feature

    def get_extra_losses(self, net_output):
        pen = []
        
        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "vq_loss" in net_output:
            pen.append(net_output["vq_loss"])

        return pen

    
