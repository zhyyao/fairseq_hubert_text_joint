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
    TransformerCrossAttnSentenceEncoderLayer
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
import time
from fairseq.modules.transformer_sentence_encoder import init_bert_params

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
class HubertCrossAttnMtlConfig(HubertConfig,MaskedTextEncoderConfig):
    shared_encoder_layer: int = field(
        default=1,
        metadata={"help": "the number of shared encoder layers"},
    )
    swap_embedding_ratio: float = field(
        default=0.3,
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
    add_ltr_layer: bool = field(
        default=False,
        metadata={
            "help":(
                "ltr 13 layer"
            )
        }
    )
    add_phf_loss_paired: bool = field(
        default=False,
        metadata={
            "help":(
                "ltr 13 layer"
            )
        }
    )
    add_ctc_after_num_updates: int= field(
        default = 0,
        metadata={
            "help":(
                "add char ctc loss after X num updates"
            )
        }
    )
    remove_phf_loss: bool = field(
        default = False,
        metadata={
            "help":(
                "remove phf loss"
            )
        }
    )
    no_extra_ltr_layer: bool = field(
        default=False,
        metadata={
            "help":(
                "no extra ltr layer"
            )
        }
    )
    paired_ce: bool = field(
        default=False,
        metadata={
            "help":(
                "paired ce loss"
            )
        }
    )


class TransformerCrossEncoder(nn.Module):
    def __init__(
        self,
        speech_encoder,
        text_encoder=None,
        shared_encoder=None,
        ltr_encoder=None
    ):
        super().__init__()
        self.speech_encoder = speech_encoder
        self.text_encoder = text_encoder
        self.shared_encoder = shared_encoder
        self.ltr_encoder = ltr_encoder

        self.apply(init_bert_params)

@register_model("hubert_cross_attn_mtl", dataclass=HubertCrossAttnMtlConfig)
class HubertCrossAttnMtlModel(BaseFairseqModel):
    def __init__(
        self,
        cfg :HubertCrossAttnMtlConfig ,
        speech_encoder : BaseFairseqModel,
        text_encoder : BaseFairseqModel,
        phf_dictionary : Dictionary,
        target_dictionary : Dictionary
    ):
        super().__init__()
        encoder_embed_dim = cfg.encoder_embed_dim
        self.remove_phf_loss = cfg.remove_phf_loss

        self.add_ltr_layer = cfg.add_ltr_layer
        self.add_phf_loss_paired = cfg.add_phf_loss_paired
        self.add_ctc_after_num_updates = cfg.add_ctc_after_num_updates
        self.speech_encoder = speech_encoder
        self.text_encoder = text_encoder
        self.no_extra_ltr_layer = cfg.no_extra_ltr_layer
        self.paired_ce = cfg.paired_ce
        shared_encoder = nn.ModuleList(
                [
                    TransformerCrossAttnSentenceEncoderLayer(
                        embedding_dim=encoder_embed_dim,
                        ffn_embedding_dim=speech_encoder.encoder.args.encoder_ffn_embed_dim,
                        num_attention_heads=speech_encoder.encoder.args.encoder_attention_heads,
                        dropout=speech_encoder.encoder.dropout,
                        attention_dropout=speech_encoder.encoder.args.attention_dropout,
                        activation_dropout=speech_encoder.encoder.args.activation_dropout,
                        activation_fn=speech_encoder.encoder.args.activation_fn,
                        layer_norm_first=speech_encoder.encoder.args.layer_norm_first,
                        has_relative_attention_bias=False,
                        num_buckets=speech_encoder.encoder.num_buckets,
                        max_distance=speech_encoder.encoder.max_distance,
                        fp32_attention=False,
                        gru_rel_pos=speech_encoder.encoder.args.gru_rel_pos,
                        expand_attention_head_size=speech_encoder.encoder.args.expand_attention_head_size,
                        conformer_module=False
                    )
                    for i in range(cfg.shared_encoder_layer)
                ]
            )
        if self.add_ltr_layer and not self.no_extra_ltr_layer:
            ltr_encoder = TransformerCrossAttnSentenceEncoderLayer(
                            embedding_dim=encoder_embed_dim,
                            ffn_embedding_dim=speech_encoder.encoder.args.encoder_ffn_embed_dim,
                            num_attention_heads=speech_encoder.encoder.args.encoder_attention_heads,
                            dropout=speech_encoder.encoder.dropout,
                            attention_dropout=speech_encoder.encoder.args.attention_dropout,
                            activation_dropout=speech_encoder.encoder.args.activation_dropout,
                            activation_fn=speech_encoder.encoder.args.activation_fn,
                            layer_norm_first=speech_encoder.encoder.args.layer_norm_first,
                            has_relative_attention_bias=False,
                            num_buckets=speech_encoder.encoder.num_buckets,
                            max_distance=speech_encoder.encoder.max_distance,
                            fp32_attention=False,
                            gru_rel_pos=speech_encoder.encoder.args.gru_rel_pos,
                            expand_attention_head_size=speech_encoder.encoder.args.expand_attention_head_size,
                            conformer_module=False
                        )
        else: 
            ltr_encoder = None
        
        encoder_after_init = TransformerCrossEncoder(speech_encoder.encoder, text_encoder, shared_encoder,ltr_encoder)
        self.speech_encoder = speech_encoder
        self.speech_encoder.encoder = encoder_after_init.speech_encoder

        self.text_encoder = encoder_after_init.text_encoder
        self.shared_encoder = encoder_after_init.shared_encoder
        self.ltr_encoder = encoder_after_init.ltr_encoder

        self.trg_proj = Linear(encoder_embed_dim, len(target_dictionary))
        self.phf_proj = Linear(encoder_embed_dim, len(phf_dictionary))
        self.shared_encoder_layer = cfg.shared_encoder_layer
        self.swap_embedding_ratio = cfg.swap_embedding_ratio
        self.swap_embedding_phoneme_aware = cfg.swap_embedding_phoneme_aware
        self.speech_encoder_noshared_layers = (cfg.encoder_layers)
        self.speech_layer_norm_before_shared =  LayerNorm(encoder_embed_dim, elementwise_affine=False)
        self.text_layer_norm_before_shared = LayerNorm(encoder_embed_dim, elementwise_affine=False)
        # self.shared_encoder = self.speech_encoder.encoder.layers[ (cfg.encoder_layers - cfg.shared_encoder_layer): ]
        self.phf_dictionary = phf_dictionary
        
        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.paired_apply_mask = cfg.apply_mask
        self.speech_cross_attn_layers = cfg.speech_cross_attn_layers
        self.text_cross_attn_layers = cfg.text_cross_attn_layers
        self.shared_encoder_layernorm = LayerNorm(encoder_embed_dim, elementwise_affine=False )

        self.pooler_activation = utils.get_activation_fn(cfg.pooler_activation_fn)

        self.lm_head_transform_weight = nn.Linear(
            encoder_embed_dim, encoder_embed_dim
        )
        self.activation_fn = utils.get_activation_fn(cfg.activation_fn)
        self.layer_norm = LayerNorm(encoder_embed_dim)
        self.phone_padding_idx = phf_dictionary.pad()
        if self.add_ltr_layer:
            self.ltr_lm_head_transform_weight = nn.Linear(
                encoder_embed_dim, encoder_embed_dim
            )
            self.ltr_activation_fn = utils.get_activation_fn(cfg.activation_fn)
            self.ltr_layer_norm = LayerNorm(encoder_embed_dim)
    

    @classmethod
    def build_model(cls, cfg:HubertCrossAttnMtlConfig, task):
        speech_encoder = HubertModel.build_model(cfg, task)
        #if task.cfg.only_speech:
        #    text_encoder = None
        #else:
        text_encoder = MaskedTextEncoder.build_model(cfg, task)
        model = cls(cfg,speech_encoder,text_encoder, task.phoneme_dictionary, task.target_dictionary )
        return model
        
    def forward_shared_encoder(self, x, x_dict, cross_attn_layers):
        
        x = x.transpose(0,1)
        # print(len(x_dict["layer_results"]))
        pos_bias = x_dict["layer_results"][-1][2]
        for cross, layer in zip(cross_attn_layers, self.shared_encoder):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.speech_encoder.encoder.layerdrop):
                if cross == -1:
                    x, layer_attn, pos_bias, _ = layer(
                        x,
                        None,
                        x_dict["padding_mask"],
                        self_attn_padding_mask = x_dict["padding_mask"],
                        pos_bias = x_dict["layer_results"][-1][2]
                    )
                else :
                    x, layer_attn, pos_bias, _ = layer(
                        x,
                        x_dict["layer_results"][cross][0],
                        x_dict["padding_mask"],
                        self_attn_padding_mask = x_dict["padding_mask"],
                        pos_bias = x_dict["layer_results"][-1][2]
                    )
        x = x.transpose(0,1)
        return x, pos_bias
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
        
        s_t = time.time()
        x_dict = self.speech_encoder(
            source,
            target_list,
            padding_mask,
            boundary,
            mask,
            features_only=True,
            output_layer=self.speech_encoder_noshared_layers,
        )
        x = x_dict["x"]
        x,pos_bias = self.forward_shared_encoder(x,x_dict, self.speech_cross_attn_layers)
        if features_only:
            return {
                "x": x,
                "padding_mask": x_dict["padding_mask"], 
                "features": x_dict["features"], 
                "layer_results": x_dict["layer_results"], 
                "mask_indices": x_dict["mask_indices"],
                "target_list": x_dict["target_list"], 
                "features_pen": x_dict["features_pen"]
            }
        r = self.speech_encoder.hubert_process(
            x,
            x_dict["padding_mask"],
            x_dict["mask_indices"],
            x_dict["target_list"],
            x_dict["features_pen"]
        )
        r["time_speech"] = time.time() - s_t
        r["phf_out"] = None
        r["trg_out"] = None
        return r

    def forward_text_only(
        self,
        prev_phoneme,
        prev_phoneme_mask=None,
        apply_mask=True,
        pos_bias = None,
        masked_tokens = None
    ):
        s_t = time.time()
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
        # x = self.text_layer_norm_before_shared(x)
        x, _ = self.forward_shared_encoder(x, text_out, self.text_cross_attn_layers)
        ltr_x = None
        if self.add_ltr_layer and  self.num_update> self.add_ctc_after_num_updates and self.no_extra_ltr_layer:
            ltr_x = x
        if self.add_ltr_layer and  self.num_update> self.add_ctc_after_num_updates and not self.no_extra_ltr_layer:
            x = x.transpose(0,1)
            ltr_x, layer_attn, pos_bias, _ = self.ltr_encoder(
                x,
                None,
                text_out["padding_mask"],
                self_attn_padding_mask = text_out["padding_mask"],
                pos_bias = text_out["layer_results"][-1][2] 
            )
            ltr_x = ltr_x.transpose(0,1)
            x = x.transpose(0,1)
        if masked_tokens is not None:
            x = x[masked_tokens,:]
        if not self.remove_phf_loss:
            x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))
            x = self.phf_proj(x)
        else:
            x = None
        if self.add_ltr_layer and self.num_update > self.add_ctc_after_num_updates:
            ltr_x = self.ltr_layer_norm(self.ltr_activation_fn(self.ltr_lm_head_transform_weight(ltr_x)))
            ltr_x = self.trg_proj(ltr_x)
        # x = x.transpose(0,1)
        # for layer in self.shared_encoder:
        #     x,_ ,pos_bias = layer(x, self_attn_padding_mask=padding_mask,  pos_bias=pos_bias)
        # x = x.transpose(0,1)
        # x = self.trg_proj(x)
        
        return {
            "phf_out": x,
            "trg_out": ltr_x,
            "phoneme_padding_mask": padding_mask,
            "time_text": time.time() - s_t,
        }

    def set_num_updates(self,num_update):
        self.num_update = num_update

    def forward_paired_data(
        self, 
        source,
        phf,
        phf_padding_mask,
        accum_list,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        boundary: Optional[List[int]] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
        finetune: bool = False
    ):
        # here use mask is for specaugment
        # down the mask prob 
        s_t = time.time()
        x_dict = self.speech_encoder(
            source,
            target_list,
            padding_mask,
            boundary,
            mask,
            features_only=True,
            output_layer=self.speech_encoder_noshared_layers,
        )
        
    
        padding_mask = x_dict["padding_mask"]
        pos_bias = x_dict["layer_results"][-1][2]
        x=x_dict["x"]
        if phf is not None:            
            # downsample 
            accum_list = accum_list
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
            if not self.paired_ce :
                x,xt = self.swap_embedding(x, xt, accum_list, x_dict["mask_indices"])
                pair_loss = None

            else:
                pair_loss = (x,xt)
        
        x,pos_bias = self.forward_shared_encoder(x,x_dict, self.speech_cross_attn_layers)
        xt = None
        #if self.add_phf_loss_paired:
        #    xt = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))
        #    xt = self.phf_proj(x)
        #if self.add_ltr_layer and finetune:
        #    x = x.transpose(0,1)
        #    x, layer_attn, pos_bias, _ = self.ltr_encoder(
        #        x,
        #        None,
        #        text_out["padding_mask"],
        #        self_attn_padding_mask = text_out["padding_mask"],
        #        pos_bias = text_out["layer_results"][-1][2] 
        #    )
        #    x = x.transpose(0,1)
        #    # x = self.ltr_layer_norm(self.ltr_activation_fn(self.ltr_lm_head_transform_weight(x)))
        #    # x = self.trg_proj(x)
        if not features_only:
            r = self.speech_encoder.hubert_process(
                x,
                x_dict["padding_mask"],
                x_dict["mask_indices"],
                x_dict["target_list"],
                x_dict["features_pen"]
            )
        r["time_paired"] = time.time() - s_t
        r["phf_out"] = xt
        r["trg_out"] = None
        r["pair_loss"] = pair_loss
        return r

    def forward(
        self,
        mode: str, 
        source: torch.Tensor=None,
        phf_input=None,
        phf_padding_mask=None,
        accum_list = None,
        lengths=None,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        boundary: Optional[List[int]] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
        masked_tokens = None,
        finetune:bool = False 
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
            return self.forward_paired_data(
                source,
                phf_input,
                phf_padding_mask,
                accum_list,
                target_list,
                padding_mask,
                boundary,
                mask,
                features_only,
                output_layer,
                finetune
            )
        elif  mode == "text_only":
            return self.forward_text_only(phf_input, phf_padding_mask,masked_tokens=masked_tokens)
        else:
            raise Exception(f"unkown type {mode}")
    def swap_embedding(self,audio_embedding, text_embedding, accum_alignment, mask_indices):
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

                if math.ceil(len(accum_alignment[i])* self.swap_embedding_ratio) >= len(accum_alignment[i]) :
                    indices = list(range(1,len(accum_alignment[i])))
                else:
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
            mask[mask_indices[i]] = 1
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
        self.trg_proj = None
        self.phf_proj = None
        self.text_layer_norm_before_shared = None
        self.final_dropout = None
        self.pooler_activation = None
        self.lm_head_transform_weight = None
        self.activation_fn = None
        self.layer_norm = None
        self.speech_encoder.remove_pretraining_modules()
        self.add_phf_loss_paired = False

    def get_normalized_probs(self, logits, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1) 
