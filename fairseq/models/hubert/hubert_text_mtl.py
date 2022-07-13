# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from _typeshed import Self
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
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.tasks import FairseqTask
from omegaconf import II, MISSING
from fairseq.modules import transformer_layer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.distributed import fsdp_wrap
from fairseq.models.masked_lm import MaskedLMEncoder
from fairseq.tasks.optimize_ali_speech_language import OptimizingAlignmentTask
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models.wav2vec.wav2vec2 import TransformerSentenceEncoderLayer
from fairseq.models.hubert.hubert_asr import HubertAsrConfig, HubertCtcConfig, HubertSeq2SeqConfig


class HubertEncoder(FairseqEncoder):
    def __init__(self, cfg: HubertAsrConfig, task, tgt_dict=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }



        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )
        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        
        task = tasks.setup_task(w2v_args.task)
        if state is not None and "task_state" in state:
            # This will load the stored "dictionaries" object
            task.load_state_dict(state["task_state"])
        model = task.build_model(w2v_args.model)
        model.remove_pretraining_modules()


        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            model.load_state_dict(state["model"], strict=False)


        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        if hasattr(model, 'speech_encoder'):
            self.w2v_model = model.speech_encoder
        else:
            self.w2v_model = model
            
        self.w2v_model.remove_pretraining_modules()
        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():

            res  = self.w2v_model.forward(**w2v_args)
            x = res["x"]
            padding_mask = res["padding_mask"]
            pos_bias = res["layer_result"][-1][2]
            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)

        # if self.proj:
        #     x_ctc = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
            "pos_bias": pos_bias
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

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
class HubertTextMTLConfig(HubertCtcConfig):
    # text
    text_encoder_layers: int = field(
        default=4, metadata={"help": "num encoder layers in the text encoder"}
    )
    
    # masking
    text_encoder_apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    text_encoder_mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    text_encoder_mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
            "(normalized by length)"
        },
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
    # model
    shared_encoder_layer: int = field(
        default=1,
        metadata={"help": "the number of shared encoder layers"},
    )
    embedding_aligner_dim: int = field(
        default=1024,
        metadata={"help": "the dimension of embedding aligner"}
    )
    swap_embedding_ratio: float = field(
        default=0.2,
        metadata={"help": "the probability of embedding swapping"}
    )
    swap_embedding_phoneme_aware: bool = field(
        default=True,
        metadata={"help": "swap embedding with phoneme aware"}
    )

class MaskedTextEncoder(BaseFairseqModel):
    def __init__(
        self,
        cfg:HubertTextMTLConfig,
        dictionary,
        
    ):
        super().__init__()
        # 1. token embedding
        self.encoder_embed_dim= cfg.encoder_embed_dim
        self.token_embedding = self.build_embedding(cfg,dictionary,self.encoder_embed_dim)
        # 2. text encoder
        self.MASK = 0
        self.embedding_dim = self.encoder_embed_dim
        self.dropout = cfg.dropout
        self.encoder_layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=cfg.encoder_ffn_embed_dim,
                    num_attention_heads=cfg.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=cfg.attention_dropout,
                    activation_dropout=cfg.activation_dropout,
                    activation_fn=cfg.activation_fn,
                    layer_norm_first=cfg.layer_norm_first,
                )
                for _ in range(cfg.text_encoder_layers)
            ]
        )
        self._dictionary = dictionary
        self._mask_prob = cfg.text_encoder_mask_prob
        self._apply_mask = cfg.text_encoder_apply_mask
        self._mask_length = cfg.text_encoder_mask_length
        self._mask_selection = cfg.text_encoder_mask_selection
        self._mask_other = cfg.text_encoder_mask_other
        self._no_mask_overlap = cfg.text_encoder_no_mask_overlap
        self._mask_min_space = cfg.text_encoder_mask_min_space

    def forward(
        self,
        prev_phoneme,
        prev_phoneme_mask,
        apply_mask,
    ):
        # 1. apply mask
        
        if apply_mask:
            prev_phoneme, _ = self.apply_mask(prev_phoneme, prev_phoneme_mask, self._dictionary)
        # 2. embedding
        prev_phoneme = self.token_embedding(prev_phoneme)
        prev_phoneme = prev_phoneme.transpose(0,1)
        # 3. encoder
        pos_bias=None
        for transformer in self.encoder_layers:
            prev_phoneme,_,pos_bias = transformer(prev_phoneme, self_attn_padding_mask=prev_phoneme_mask, pos_bias=pos_bias)
        prev_phoneme = prev_phoneme.transpose(0,1)
        return {
            "encoder_out": prev_phoneme,
            "padding_mask": prev_phoneme_mask,
            "pos_bias": pos_bias
        }
    
    def apply_mask(self, x, padding_mask, target_list):
        B, T = x.shape
        if self._mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self._mask_prob,
                self._mask_length,
                self._mask_selection,
                self._mask_other,
                min_masks=2,
                no_overlap=self._no_mask_overlap,
                min_space=self._mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.MASK
        else:
            mask_indices = None
        return x, mask_indices
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


    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

@register_model("hubert_text_mtl", dataclass=HubertTextMTLConfig)
class HubertTextMTL(BaseFairseqModel):
    def __init__(
        self, 
        cfg: HubertTextMTLConfig, 
        w2v_encoder: BaseFairseqModel, 
        text_encoder:BaseFairseqModel, 
        embedding_aligner,
        has_shared_encoder
    ):
        super().__init__()
        self.cfg = cfg
        arg_overrides = cfg.w2v_args.model.copy()
        arg_overrides["encoder_layers"] = cfg.shared_encoder_layer
        # 1. audio encoder
        self.w2v_encoder = w2v_encoder
        # 2. text encoder
        self.text_encoder = text_encoder
        # 4. shared encoder
        self.embedding_dim = cfg.w2v_args.model.encoder_embed_dim
        self.dropout = cfg.w2v_args.model.dropout
        if has_shared_encoder:
            self.shared_encoder = self.w2v_encoder.w2v_model.encoder.layers[-(self.cfg.shared_encoder_layer):]
        else:
            self.shared_encoder = None
        # 5. embedding aligner
        self.embedding_aligner = embedding_aligner
        # 6. ctc proj
        self.swap_embedding_ratio = cfg.swap_embedding_ratio
        self.swap_embedding_phoneme_aware = cfg.swap_embedding_phoneme_aware
        
        
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

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
    def build_model(cls, cfg: HubertTextMTLConfig, task: FairseqTask):
        """Build a new model instance."""
        # 1. audio encoder
        w2v_encoder = HubertEncoder(cfg, task, task.target_dictionary)
        if task.has_text_encoder:
            # 2. text encoder
            text_encoder = MaskedTextEncoder(
                cfg,
                task.phoneme_dictionary
            )
        else:
            text_encoder = None
        # embedding_aligner
        embedding_aligner = nn.parameter.Parameter(
            nn.init.uniform_(torch.empty(
                (len(task.phoneme_dictionary), cfg.w2v_args.model.encoder_embed_dim)
            ))
        )
        
        return cls(cfg, w2v_encoder, text_encoder, embedding_aligner, task.has_shared_encoder)

    def forward(
        self, 
        source, 
        padding_mask,
        phf_input,
        phf_padding_mask,
        lengths,
        mode: str = "speech",
    ):
        if mode == "speech":
            return self.forward_speech(
                source,
                padding_mask,
                phf_input,
                phf_padding_mask
            )
        elif mode == "text":
            return self.forward_text(
                phf_input,
                phf_padding_mask
            )

    def swap_embedding(self,audio_embedding, text_embedding, accum_alignment):
        # audio_embedding is B*T*D 
        # text_embedding is B*T*D
        # assert the length of audio embedding is the same as text embedding
        assert(audio_embedding.shape[1] == text_embedding.shape[1]), str((audio_embedding.shape,text_embedding.shape))
        # building mask
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
            


    def get_accum_from_phoneme_seq(self, phoneme_seq, phoneme_padding_mask):
        bsz = phoneme_seq.shape[0]
        accum_lists = []
        for i in range(bsz):
            accum = [indice+1 for indice,j in enumerate(range(phoneme_seq[i].shape[0]-1)) 
                if phoneme_padding_mask[i][j] == False and phoneme_seq[i][j]!=phoneme_seq[i][j+1] ]
            accum_lists.append(accum)
        return accum_lists
    
    def forward_speech(
        self,
        x,
        padding_mask,
        xt,
        phoneme_padding_mask
    ):
        # 1. audio encoder
        # assert audio input is feature
        assert(len(x.shape)==2)
        
        x_dict = self.w2v_encoder(x, padding_mask, False, output_layer = (len(self.w2v_encoder.w2v_model.encoder.layers) - self.cfg.shared_encoder_layer) )
        # x = x_dict["encoder_out"]
        if self.text_encoder is not None and xt is not None and self.training:
            # because of w2v downsample we do downsample here
            '''
                notice!!!!! 
                if we change the structure of w2v conv, we should change the downsample here
                this downsample is compute as follow:
                    w2v conv structure [(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2
                the field we can seen is
                    { (x[i]-1)*2+2 = x[i-1] }*2
                    { (x[i]-1)*2+3 = x[i-1] }*4
                    { (x[i]-1)*5+10 = x[i-1] }*1
                then we get 400 which is the same as feature extraction of fbank
                then we can compute overlap from the top (512,3,2)
                    { (x[i]-1)*2+3 = x[i-1] }*4
                    { (x[i]-1)*5+10 = x[i-1] }*1
                finally we get 80 and ( 400 - 80 ) = 320 = 160 * 2,
                so the conv structure can map to the kaldi fbank feature by the kaldi fbank
                feature downsampling of twice
            '''
            xt = xt[:,::2]
            phoneme_padding_mask = phoneme_padding_mask[:,::2]
            # 2. text_encoder 
            accum_list = self.get_accum_from_phoneme_seq(xt, phoneme_padding_mask)
            # for swapping embedding we do not mask the input
            xt_dict = self.text_encoder(xt,phoneme_padding_mask, apply_mask=False )
            # 3. text_encoder -> swap embedding
            assert(x_dict["encoder_out"].shape[1] == xt_dict["encoder_out"].shape[1]), str((x_dict["encoder_out"].shape,xt_dict["encoder_out"].shape))+ ", "+ str(x.shape)+ ", "+str(xt.shape)
            self.swap_embedding(
                x_dict["encoder_out"], 
                xt_dict["encoder_out"],
                accum_list
            )
        # 4. audio encoder -> embedding aligner -> ctc prob
        #    text encoder -> embedding aligner -> mlm prob
        x = x_dict["encoder_out"]
        pos_bias = x_dict["pos_bias"]
        x_out = None
        if self.shared_encoder:
            x_out = utils.log_softmax(torch.cdist(x,self.embedding_aligner), -1)
            x = x.transpose(0,1)
            # 5. audio encoder -> shared encoder
            
            for transformer in self.shared_encoder:
                x,_,pos_bias = transformer(x, self_attn_padding_mask=x_dict["encoder_padding_mask"],pos_bias=pos_bias)
            x = x.transpose(0,1)
        x = self.w2v_encoder.proj(x)
        return {
            "ctc_prob": x_out,
            "final_ctc_prob": x,
            "padding_mask": x_dict["encoder_padding_mask"],
        }


    def forward_text(
        self,
        prev_phoneme,
        phoneme_padding_mask
    ):
        # 1. text encoder
        out_dict = self.text_encoder(prev_phoneme, phoneme_padding_mask, apply_mask=True)
        phoneme_padding_mask = out_dict["padding_mask"]
        # 2. audio encoder -> embedding aligner -> MLM prob
        # x = out_dict["encoder_out"]
        
        # 4. audio encoder -> shared encoder
        out = out_dict["encoder_out"]
        out = out.transpose(0,1)
        pos_bias=out_dict["pos_bias"]
        for transformer in self.shared_encoder:
            out,_,pos_bias = transformer(out, self_attn_padding_mask=phoneme_padding_mask,pos_bias=pos_bias)
        out = out.transpose(0,1)
        x = torch.cdist(out,self.embedding_aligner)
        out = self.w2v_encoder.proj(out)
        return {
            "mlm_prob": x,
            "final_ctc_prob": out,
            "phoneme_padding_mask": phoneme_padding_mask
        }

    def get_normalized_probs(self, net_output,name, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output[name].transpose(0,1)
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

