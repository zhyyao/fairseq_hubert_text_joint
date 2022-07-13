# v-zhuoyao update
import contextlib
from argparse import Namespace
from logging import setLogRecordFactory
import math
from typing import Any, List
import random
from fairseq.data.dictionary import Dictionary

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.tasks import FairseqTask
from omegaconf import II, MISSING
from fairseq.modules import transformer_layer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.distributed import fsdp_wrap
from fairseq.models.masked_lm import MaskedLMEncoder
from fairseq.tasks.joint_hubert_mlm_pretrain import JointHubertMlmPretrainTask
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.modules.transformer_sentence_encoder import init_bert_params


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
class MaskedTextEncoderConfig(FairseqDataclass):
    # text
    text_encoder_layers: int = field(
        default=3, metadata={"help": "num encoder layers in the text encoder"}
    )
    
    # masking
    text_encoder_apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    text_encoder_mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    
    text_encoder_mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    text_encoder_mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    text_encoder_no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    text_encoder_mask_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

class MaskedTextEncoder(TransformerEncoder):
    def __init__(
        self,
        cfg:MaskedTextEncoderConfig,
        dictionary,
    ):
        cfg_text = cfg.copy()
        cfg_text["encoder_layers"] = cfg.text_encoder_layers
        super().__init__(cfg_text)
        # 1. token embedding
        self.encoder_embed_dim= cfg.encoder_embed_dim
        self.token_embedding = self.build_embedding(cfg,dictionary,self.encoder_embed_dim)
        # 2. text encoder
        self.embedding_dim = self.encoder_embed_dim
        self.dropout = cfg.dropout
        self._dictionary = dictionary
        # self._mask_prob = cfg.text_encoder_mask_prob
        self._apply_mask = cfg.text_encoder_apply_mask
        self._mask_length = cfg.text_encoder_mask_length
        self._mask_selection = cfg.text_encoder_mask_selection
        self._mask_other = cfg.text_encoder_mask_other
        self._no_mask_overlap = cfg.text_encoder_no_mask_overlap
        self._mask_min_space = cfg.text_encoder_mask_min_space
        self.text_layer_norm =  LayerNorm(self.encoder_embed_dim, elementwise_affine=False)


    def forward(
        self,
        prev_phoneme,
        prev_phoneme_mask,
        apply_mask,
        pos_bias = None
    ):
        # 1. apply mask
        
        # if apply_mask:
        #     prev_phoneme, _ = self.apply_mask(prev_phoneme, prev_phoneme_mask)
        # 2. embedding
        prev_phoneme = self.token_embedding(prev_phoneme)
        # prev_phoneme = self.text_layer_norm(prev_phoneme)
        prev_phoneme, layer_results = self.extract_features(
            prev_phoneme, padding_mask=prev_phoneme_mask,
            tgt_layer=len(self.layers)-1,
            pos_bias=pos_bias,
        )
        assert len(layer_results) == len(self.layers), "len layer_result:"+ str(len(layer_results)) + ", len layer" + str(len(self.layers))
        return {
            "encoder_out": prev_phoneme,
            "padding_mask": prev_phoneme_mask,
            "pos_bias": layer_results[-1][2],
            "layer_results" : layer_results
        }
    

    # def apply_mask(self, x, padding_mask):
    #     B, T = x.shape
    #     if self._mask_prob > 0:
    #         mask_indices = compute_mask_indices(
    #             (B, T),
    #             padding_mask,
    #             self._mask_prob,
    #             self._mask_length,
    #             self._mask_selection,
    #             self._mask_other,
    #             min_masks=2,
    #             no_overlap=self._no_mask_overlap,
    #             min_space=self._mask_min_space,
    #         )
    #         mask_indices = torch.from_numpy(mask_indices).to(x.device)
    #         x[mask_indices] = self.MASK
    #     else:
    #         mask_indices = None
    #     return x, mask_indices
    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb
    
    @classmethod
    def build_model(cls,cfg: MaskedTextEncoderConfig, task:JointHubertMlmPretrainTask):
        
        model = MaskedTextEncoder(cfg, task.phoneme_dictionary)
        return model
