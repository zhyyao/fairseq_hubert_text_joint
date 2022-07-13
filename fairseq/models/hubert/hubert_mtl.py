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
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    ConvFeatureExtractionModel,
    TransformerEncoder,
)
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.tasks.hubert_pretraining import (
    HubertPretrainingConfig,
    HubertPretrainingTask,
)
from fairseq.tasks.joint_hubert_mlm_pretrain import (
    JointSpeechTextPretrainConfig,
    JointHubertMlmPretrainTask
)
from fairseq.models.hubert.hubert_text_mtl import MaskedTextEncoder, HubertTextMTLConfig
from fairseq.models.hubert.hubert import HubertConfig, HubertModel
from fairseq.models.hubert.masked_text import MaskedTextEncoderConfig, MaskedTextEncoder
from omegaconf import II
import random
import math

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
class HubertMtlConfig(HubertConfig,MaskedTextEncoderConfig):
    shared_encoder_layer: int = field(
        default=1,
        metadata={"help": "the number of shared encoder layers"},
    )
    swap_embedding_ratio: float = field(
        default=0.2,
        metadata={"help": "the probability of embedding swapping"}
    )
    swap_embedding_phoneme_aware: bool = field(
        default=True,
        metadata={"help": "swap embedding with phoneme aware"}
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

@register_model("hubert_mtl", dataclass=HubertMtlConfig)
class HubertMtlModel(BaseFairseqModel):
    def __init__(
        self,
        cfg :HubertMtlConfig ,
        speech_encoder : BaseFairseqModel,
        text_encoder : BaseFairseqModel,
        phf_dictionary : Dictionary,
        target_dictionary : Dictionary
    ):
        super().__init__()
        encoder_embed_dim = cfg.encoder_embed_dim
        self.speech_encoder = speech_encoder
        self.text_encoder = text_encoder
        self.trg_proj = Linear(encoder_embed_dim, len(target_dictionary))
        self.phf_proj = Linear(encoder_embed_dim, len(phf_dictionary))
        self.shared_encoder_layer = cfg.shared_encoder_layer
        self.swap_embedding_ratio = cfg.swap_embedding_ratio
        self.swap_embedding_phoneme_aware = cfg.swap_embedding_phoneme_aware
        self.speech_encoder_noshared_layers = (cfg.encoder_layers - cfg.shared_encoder_layer)
        self.speech_layer_norm_before_shared =  LayerNorm(encoder_embed_dim, elementwise_affine=False)
        self.text_layer_norm_before_shared = LayerNorm(encoder_embed_dim, elementwise_affine=False)
        self.shared_encoder = self.speech_encoder.encoder.layers[ (cfg.encoder_layers - cfg.shared_encoder_layer): ]
        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.paired_apply_mask = cfg.apply_mask

        
    @classmethod
    def build_model(cls, cfg:HubertMtlConfig, task:JointHubertMlmPretrainTask):
        speech_encoder = HubertModel.build_model(cfg, task)
        text_encoder = MaskedTextEncoder.build_model(cfg, task)
        model = cls(cfg,speech_encoder,text_encoder, task.phoneme_dictionary, task.target_dictionary )
        return model
        
    def forward_speech_only(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        boundary: Optional[List[int]] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ):
        return self.speech_encoder(
            source,
            target_list,
            padding_mask,
            boundary,
            mask,
            features_only,
            output_layer,
            layer_norm_before_shared=self.speech_encoder_noshared_layers
        )
    def forward_text_only(
        self,
        prev_phoneme,
        prev_phoneme_mask,
        apply_mask=True,
        pos_bias = None
    ):
        text_out = self.text_encoder(
            prev_phoneme,
            prev_phoneme_mask,
            apply_mask,
            pos_bias
        )
        x = text_out["encoder_out"]
        padding_mask = text_out["padding_mask"]
        pos_bias = text_out["pos_bias"]
        x_phf = self.phf_proj(x)
        x = self.text_layer_norm_before_shared(x)
        x = x.transpose(0,1)
        for layer in self.shared_encoder:
            x,_ ,pos_bias = layer(x, self_attn_padding_mask=padding_mask,  pos_bias=pos_bias)
        x = x.transpose(0,1)
        x = self.trg_proj(x)
        return {
            "phf_out": x_phf,
            "trg_out": x,
            "phoneme_padding_mask": padding_mask,
        }

    def forward_paired_data(
        self, 
        source,
        padding_mask,
        phf,
        phf_padding_mask,
    ):
        # here use mask is for specaugment
        # down the mask prob 
        src_mask = self.speech_encoder.mask_prob
        self.speech_encoder.mask_prob = 0.5
        x = self.speech_encoder(
            source,
            padding_mask=padding_mask,
            mask=self.paired_apply_mask,
            features_only=True,
            output_layer=self.speech_encoder_noshared_layers
        )
        self.speech_encoder.mask_prob = src_mask
        assert self.speech_encoder_noshared_layers == len(x["layer_results"]), "speech layers: "+ \
            str(self.speech_encoder_noshared_layers)+", layer_result: "+ str(len(x["layer_results"]))
    
        padding_mask = x["padding_mask"]
        pos_bias = x["layer_results"][-1][2]
        x=x["x"]
        if phf is not None:
            phf_prob = self.phf_proj(x)
            
            # downsample 
            phf = phf[:,::2]
            phf_padding_mask = phf_padding_mask[:,::2]
            accum_list = self.get_accum_from_phoneme_seq(phf, phf_padding_mask)
            xt = self.text_encoder(
                phf,
                phf_padding_mask,
                apply_mask = False,
                pos_bias = None
            )

            phf_padding_mask = xt["padding_mask"]
            phf_pos_bias = xt["pos_bias"]
            xt = xt["encoder_out"]

            #swap embedding
            x,xt = self.swap_embedding(x, xt, accum_list)
            
        x = x.transpose(0,1)
        #if phf is not None:
        #    xt = xt.transpose(0,1)
        x = self.speech_layer_norm_before_shared(x)
        for layer in self.shared_encoder:
            x,_,pos_bias = layer(x,self_attn_padding_mask=padding_mask, pos_bias=pos_bias)
        #    if phf is not None:
        #        xt,_,phf_pos_bias = layer(xt, self_attn_padding_mask=phf_padding_mask, pos_bias=phf_pos_bias)
        x = self.final_dropout(x.transpose(0,1))
        #if phf is not None:
        #    xt = self.final_dropout(xt.transpose(0,1))
        trg_prob_x = self.trg_proj(x)
        #if phf is not None:
        #    trg_prob_xt = self.trg_proj(xt)
        #if phf is None:
        #    phf_prob = None
        #    trg_prob_xt = None
        phf_prob = None
        trg_prob_xt = None

        return {
            "phf_out": phf_prob,
            "trg_out": trg_prob_x,
            "trg_out_t":  trg_prob_xt,
            "padding_mask": [padding_mask, phf_padding_mask]
        }

    def forward(
        self,
        source: torch.Tensor,
        mode: str, 
        phf_input=None,
        phf_padding_mask=None,
        lengths=None,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        boundary: Optional[List[int]] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ):
        if mode=="speech_only":
            return self.forward_speech_only(
                source,
                target_list,
                padding_mask,
                boundary,
                mask,
                features_only,
                output_layer,
            )

        elif mode == "paired_data":
            return self.forward_paired_data(source, padding_mask, phf_input, phf_padding_mask)
        elif mode == "text_only":
            return self.forward_text_only(phf_input, phf_padding_mask)

    def swap_embedding(self,audio_embedding, text_embedding, accum_alignment):
        # audio_embedding is B*T*D 
        # text_embedding is B*T*D
        # assert the length of audio embedding is the same as text embedding
        assert(audio_embedding.shape[1] == text_embedding.shape[1]), str((audio_embedding.shape,text_embedding.shape))
        # building mask
        if self.swap_embedding_ratio == 0.0:
            return audio_embedding, text_embedding
        bsz = audio_embedding.shape[0]
        channel_size = audio_embedding.shape[2]
        for i in range(bsz):
            if self.swap_embedding_phoneme_aware:
                mask = torch.ones((audio_embedding.shape[1]), device=audio_embedding.device)

                indices = random.sample(list(range(1,len(accum_alignment[i]))), 
                    math.ceil(len(accum_alignment[i])* self.swap_embedding_ratio))
                for index in indices:
                    start,end = accum_alignment[i][index-1], accum_alignment[i][index]
                    mask[start:end] = 0
            else:
                mask = (
                    torch.randn(
                        text_embedding[i].shape[0], 
                        device=text_embedding.device
                    ).uniform() > self.swap_embedding_ratio
                ).float()
            mask = mask.unsqueeze(1).expand(mask.shape[0],channel_size)
            text_embedding_tmp = text_embedding[i] * mask + audio_embedding[i] * (1 - mask)
            audio_embedding[i] = audio_embedding[i] * mask + text_embedding[i] * (1 - mask)
            text_embedding[i] = text_embedding_tmp
        return audio_embedding, text_embedding
            


    def get_accum_from_phoneme_seq(self, phoneme_seq, phoneme_padding_mask):
        bsz = phoneme_seq.shape[0]
        accum_lists = []
        for i in range(bsz):
            accum = [indice+1 for indice,j in enumerate(range(phoneme_seq[i].shape[0]-1)) 
                if phoneme_padding_mask[i][j] == False and phoneme_seq[i][j]!=phoneme_seq[i][j+1] ]
            accum_lists.append(accum)
        return accum_lists

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def get_logits(self, net_output, is_masked=True):
        return self.speech_encoder.get_logits(net_output,is_masked)

    def get_targets(self, net_output, is_masked=True):
        return self.speech_encoder.get_targets(net_output,is_masked)
    
    def get_extra_losses(self, net_output):
        return self.speech_encoder.get_extra_losses(net_output)

    def remove_pretraining_modules(self,):
        self.text_encoder = None
        self.speech_encoder.remove_pretraining_modules()
    
    def get_normalized_probs(self, logits, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1) 
