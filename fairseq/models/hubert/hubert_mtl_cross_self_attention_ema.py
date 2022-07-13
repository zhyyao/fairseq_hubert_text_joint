# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Tuple
from torch import Tensor

import numpy as np

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq.data.dictionary import Dictionary
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    TransformerCrossAttnSentenceEncoderLayer
)
from fairseq.modules import GradMultiply, LayerNorm

from fairseq.models.hubert.hubert_text_mtl import MaskedTextEncoder, HubertTextMTLConfig
from fairseq.models.hubert.data2vec_audio import Data2VecAudioModel, Data2VecAudioConfig
from fairseq.models.hubert.masked_text import MaskedTextEncoderConfig, MaskedTextEncoder
from omegaconf import II
import math
import time

from fairseq.modules import EMAModule, EMAModuleConfig

import torch.nn.functional as F

import torch.distributed as dist

logger = logging.getLogger(__name__)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

@dataclass
class CrossAttnEMAConfig(Data2VecAudioConfig,MaskedTextEncoderConfig):
    shared_encoder_layer: int = field(
        default=1,
        metadata={"help": "the number of shared encoder layers"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout after transformer and before final projection"
        },
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in hubert"},
    )
    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    decoder_embed_dim: int  = II("model.encoder_embed_dim")
    decoder_normalize_before: bool = II("model.layer_norm_first")
    decoder_ffn_embed_dim: int  = II("model.encoder_ffn_embed_dim")
    decoder_attention_heads : int = II("model.encoder_attention_heads")
    speech_cross_attn_layers: List[int] = field(
        default_factory=lambda: [2,5,8],
        metadata={
            "help": (
                "sppech encoder cross attention layer"
            )
        },
    )
    text_cross_attn_layers: List[int] = field(
        default_factory=lambda: [2,5,8],
        metadata={
            "help": (
                "sppech encoder cross attention layer"
            )
        },
    )
    activation_fn: str = field(
        default="gelu",
        metadata={
            "help":(
                "active function"
            )
        }
    )
    pooler_activation_fn: str = field(
        default="tanh",
        metadata={
            "help":(
                "output activiation"
            )
        }
    )

def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining

@register_model("cross_attn_ema", dataclass=CrossAttnEMAConfig)
class CrossAttnEMAModel(BaseFairseqModel):
    def __init__(
        self,
        cfg :CrossAttnEMAConfig ,
        speech_encoder : BaseFairseqModel,
        text_encoder : BaseFairseqModel,
        phoneme_dictionary
    ):
        super().__init__()
        self.embed = cfg.encoder_embed_dim
        # build model architecture
        self.cfg = cfg
        self.speech_encoder = speech_encoder
        self.text_encoder = text_encoder
        self.padding_idx = phoneme_dictionary.pad()
        self.mask_idx = phoneme_dictionary.index("<mask>")
        self.phoneme_dictionary = phoneme_dictionary
        self.shared_encoder = nn.ModuleList(
                [
                    TransformerCrossAttnSentenceEncoderLayer(
                        embedding_dim=self.embed,
                        ffn_embedding_dim=self.speech_encoder.encoder.args.encoder_ffn_embed_dim,
                        num_attention_heads=self.speech_encoder.encoder.args.encoder_attention_heads,
                        dropout=self.speech_encoder.encoder.dropout,
                        attention_dropout=self.speech_encoder.encoder.args.attention_dropout,
                        activation_dropout=self.speech_encoder.encoder.args.activation_dropout,
                        activation_fn=self.speech_encoder.encoder.args.activation_fn,
                        layer_norm_first=self.speech_encoder.encoder.args.layer_norm_first,
                        has_relative_attention_bias=(self.speech_encoder.encoder.relative_position_embedding and i == 0),
                        num_buckets=self.speech_encoder.encoder.num_buckets,
                        max_distance=self.speech_encoder.encoder.max_distance,
                        fp32_attention=False,
                        gru_rel_pos=self.speech_encoder.encoder.args.gru_rel_pos,
                        expand_attention_head_size=self.speech_encoder.encoder.args.expand_attention_head_size,
                        conformer_module=False
                    )
                    for i in range(cfg.shared_encoder_layer)
                ]
            )
        self.speech_cross_attn_layers = cfg.speech_cross_attn_layers
        self.text_cross_attn_layers = cfg.text_cross_attn_layers

        # teacher model define
        self.ema = None
        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        # student final proj
        self.speech_final_proj = nn.Linear(self.embed, self.embed)
        self.text_final_proj = nn.Linear(self.embed, self.embed)
        self.step_time = None
        
        

    def make_ema_teacher(self):
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.ema_decay,
            ema_fp32=True,
        )
        skip_keys = set()
        if self.cfg.ema_layers_only:
            self.cfg.ema_transformer_only = True
            for k, _ in self.speech_encoder.feature_extractor.named_parameters():
                skip_keys.add(f"speech_encoder.feature_extractor.{k}")
            for k, _ in self.speech_encoder.post_extract_proj.named_parameters():
                skip_keys.add(f"speech_encoder.post_extract_proj.{k}")
            for k, _ in self.speech_encoder.encoder.pos_conv.named_parameters():
                skip_keys.add(f"speech_encoder.encoder.pos_conv.{k}")
            for k, _ in self.text_encoder.pos_conv.named_parameters():
                skip_keys.add(f"text_encoder.pos_conv.{k}")
            for k, _ in self.text_encoder.token_embedding.named_parameters():
                skip_keys.add(f"text_encoder.token_embedding.{k}")

        self.ema = EMAModule(
            self,
            ema_config,
            skip_keys=skip_keys,
        )

    def set_num_updates(self, num_updates):
        step_time_s = time.time()
        super().set_num_updates(num_updates)

        if self.ema is None and self.speech_final_proj is not None:
            logger.info(f"making ema teacher")
            self.make_ema_teacher()
        elif self.training and self.ema is not None:
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self)
        self.step_time = time.time() - step_time_s 
        self.num_updates = num_updates

    @classmethod
    def build_model(cls, cfg:CrossAttnEMAConfig, task):
        speech_encoder = Data2VecAudioModel.build_model(cfg, task)
        text_encoder = MaskedTextEncoder.build_model(cfg, task)
        model = cls(cfg,speech_encoder,text_encoder,task.phoneme_dictionary)
        return model
        
    def forward_shared_encoder(self, x, x_dict, cross_attn_layers):
        attn: Optional[Tensor] = None
        layer_results = []
        x = x.transpose(0,1)
        # print(len(x_dict["layer_results"]))
        for cross, layer in zip(cross_attn_layers, self.shared_encoder):
            x, layer_attn, pos_bias, lr = layer(
                x,
                x_dict["layer_results"][cross][0],
                x_dict["padding_mask"],
                self_attn_padding_mask = x_dict["padding_mask"],
                pos_bias = x_dict["layer_results"][-1][2]
            )
            layer_results.append(lr)
        
        x = x.transpose(0,1)
        return x, pos_bias, layer_results

    def forward_speech_only(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        encoder_only: bool = False,
        output_layer: Optional[int] = None,
    ):
        x_dict = self.speech_encoder(
            source,
            padding_mask,
            mask,
            layer = output_layer
        )
        x = x_dict["x"]
        layer_results = []
        if not encoder_only:
            x,pos_bias, layer_results = self.forward_shared_encoder(
                x,
                x_dict, 
                self.speech_cross_attn_layers
            )
        return {
            "x": x,
            "padding_mask": x_dict["padding_mask"],
            "layer_results": layer_results,
            "mask_indices": x_dict["mask_indices"]
        }

    def forward_text_only(
        self,
        prev_phoneme,
        prev_phoneme_mask=None,
        apply_mask=True,
        pos_bias = None,
        masked_tokens = None,
        encoder_only = False
    ):
        if prev_phoneme_mask is None:
            prev_phoneme_mask = prev_phoneme.eq(self.phone_padding_idx)

        text_out = self.text_encoder(
            prev_phoneme,
            prev_phoneme_mask,
            apply_mask,
            pos_bias
        )
        x = text_out["encoder_out"]
        padding_mask = text_out["padding_mask"]
        pos_bias = text_out["pos_bias"]
        layer_results = []
        if not encoder_only:
            x, _ , layer_results = self.forward_shared_encoder(
                x, 
                text_out, 
                self.text_cross_attn_layers
            )
        
        return {
            "x": x,
            "padding_mask": padding_mask,
            "layer_results": layer_results,
            "mask_indices": masked_tokens
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.ema is not None:
            k = prefix + "_ema"
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        mode: str, 
        source: torch.Tensor=None,
        padding_mask: Optional[torch.Tensor] = None,
        lengths=None,
        mask: bool = True,
        features_only: bool = False,
        encoder_only: bool = True,
        masked_tokens = None,
        phoneme_target: torch.Tensor = None,
    ):
        if mode=="speech_only":
            re =  self.forward_speech_only(
                source,
                padding_mask = padding_mask,
                mask = mask,
                encoder_only = encoder_only,
                output_layer = ( len(self.speech_encoder.encoder.layers) - 1 )
            )
            x = re["x"]
            mask_indices = re["mask_indices"]
            orig_padding_mask = padding_mask
        elif "text_only" in mode:
            padding_mask = source.eq(self.padding_idx)
            if mask and masked_tokens is None:
                masked_tokens = source.eq(self.mask_idx)
            re = self.forward_text_only(
                source, 
                padding_mask,
                masked_tokens=masked_tokens,
                encoder_only=encoder_only
            )
            x = re["x"]
            mask_indices = re["mask_indices"]
            orig_padding_mask = padding_mask
        else:
            raise Exception(f"Unkown type {mode}")
        if features_only:
            return re

        # result
        result = {
            "losses": {},
        }
        
        with torch.no_grad():
            self.ema.model.eval()
            # if self.cfg.ema_transformer_only:
            #     y, layer_results = self.ema.model.extract_features(
            #         pre_encoder_features,
            #         padding_mask=orig_padding_mask,
            #         extract_teacher=True
            #     )
            #     mask_indices = y["mask_indices"]
            #     y = {
            #         "x": y,
            #         "padding_mask": padding_mask,
            #         "layer_results": layer_results,
            #     }
            # else:
            if "text_only" in mode:
                source = phoneme_target
            y = self.ema.model.extract_features(
                source=source,
                padding_mask=orig_padding_mask,
                extract_teacher=True,
                mask=False,
                mode=mode
            )
            
            target_layer_results = [l for l in y["layer_results"]]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    tl.permute(1, 2, 0) for tl in target_layer_results  # TBC -> BCT
                ]
                permuted = True

            if self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]

            if self.cfg.instance_norm_target_layer:
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]

            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]

            if self.cfg.group_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-2:])
                    for tl in target_layer_results
                ]

            if self.cfg.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]
            y = sum(target_layer_results) / len(target_layer_results)

            if self.cfg.layer_norm_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])

            if self.cfg.instance_norm_targets:
                y = F.instance_norm(y.float().transpose(1, 2)).transpose(1, 2)

            if not permuted:
                y = y.transpose(0, 1)

            y = y[mask_indices]
        x = x[mask_indices]
        if mode=="speech_only":
            x = self.speech_final_proj(x)
        elif  "text_only" in mode:
            x = self.text_final_proj(x)

        sz = x.size(-1)

        if self.loss_beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(
                x.float(), y.float(), reduction="none", beta=self.loss_beta
            ).sum(dim=-1)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(sz)

        result["losses"]["regression"] = loss.sum() * scale

        if "sample_size" not in result:
            result["sample_size"] = loss.numel()

        
        with torch.no_grad():
            result["target_var"] = self.compute_var(y)
            result["pred_var"] = self.compute_var(x.float())

        if self.num_updates > 5000 and result["target_var"] < self.cfg.min_target_var:
            logger.error(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
            raise Exception(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
        if self.num_updates > 5000 and result["pred_var"] < self.cfg.min_pred_var:
            logger.error(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )
            raise Exception(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )

        if self.ema is not None:
            result["ema_decay"] = self.ema.get_decay() * 1000
        if self.step_time is not None:
            result["time_step"] = self.step_time
        return result


    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()


    def remove_pretraining_modules(self,):
        self.text_encoder = None
        self.trg_proj = None
        self.phf_proj = None
        self.text_layer_norm_before_shared = None
        self.final_dropout = None
        self.pooler_activation = None
        self.lm_head_transform_weight = None
        self.activation_fn = None
        self.layer_norm = None
        self.speech_encoder.remove_pretraining_modules()
    
    def extract_features(
        self, mode, source, padding_mask, mask=False, extract_teacher=False
    ):
        res = self.forward(
            mode,
            source,
            padding_mask,
            mask=mask,
            features_only=True,
            encoder_only=False if extract_teacher else True,
        )

        return res

    
