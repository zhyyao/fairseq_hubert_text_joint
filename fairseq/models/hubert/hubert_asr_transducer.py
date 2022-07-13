# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from argparse import Namespace
from typing import Any,Dict, List, Optional, Tuple
from torch import Tensor

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
from fairseq.models import FairseqEncoderDecoderModel

@dataclass
class HubertAsrConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to hubert model"}
    )
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout after transformer and before final projection"
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability inside hubert model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights "
            "inside hubert model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
            "inside hubert model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
            "(normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in hubert to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in hubert"},
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    # this holds the loaded hubert args
    w2v_args: Any = None
    predict_layers: str = field(
        default="[12]"
    )
    separate_ctc_layer: bool = field(
        default=False
    )


@dataclass
class HubertRNNTConfig(HubertAsrConfig):
    # prediction network
    prediction_layers: int = field(
        default=2
    )
    prediction_type: str = field(
        default="uni_lstm",
    )
    prediction_embed_dim: int = field(
        default=1280,
    )
    prediction_dropout: float = field(
        default=0.3
    )

    # joint network
    joint_func: str = field(
        default="add"
    )
    joint_layers: int = field(
        default = 1
    )
    joint_hidden_dim: int = field(
        default=2048
    )

class JointNet(BaseFairseqModel):
    def __init__(self, cfg, input_size, inner_dim, vocab_size):
        super().__init__()
        self.forward_layer = Linear(input_size, inner_dim, bias=True)
        self.tanh = nn.Tanh()
        self.project_layer = Linear(inner_dim, vocab_size, bias=True)
        self.joint_func = cfg.joint_func
    
    def build_model(cls,args,task, input_size, inner_dim):
        vocab_size = len(task.target_dictionary)
        return cls(args, input_size, inner_dim, vocab_size)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)
            t = enc_state.size(1)
            u = dec_state.size(2)
            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()
        if joint_func == "cat":
            concat_state = torch.cat((enc_state, dec_state), dim=-1)
        elif joint_func = "add":
            concat_state = torch.add(enc_state, dec_state)
        outputs = self.forward_layer(concat_state)
        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)
        return outputs

@register_model("hubert_rnnt", dataclass=HubertCtcConfig)
class HubertRNNT(BaseFairseqModel):
    def __init__(self, cfg: HubertRNNTConfig, w2v_encoder: BaseFairseqModel, joint_net: JointNet):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder
        self.text_embedding = self.build_embedding(cfg,dictionary,self.prediction_embed_dim)
        self.prediction = nn.LSTM(
            input_size=cfg.prediction_embed_dim,
            hidden_size=cfg.prediction_embed_dim,
            num_layers=cfg.predict_layers,
            dropout=cfg.prediction_dropout,
            proj_size=self.w2v_encoder.w2v_model.embed
        )
        self.joint_model = joint_net

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: HubertRNNTConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = HubertEncoder(cfg, task.target_dictionary)
        joint_model = JointNet(
            cfg,
            task,
            input_size=self.w2v_encoder.w2v_model.embed, 
            inner_dim=cfg.joint_hidden_dim
        )
        return cls(cfg, w2v_encoder, joint_net)

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

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][..., 0] = 0
            logits[padding][..., 1:] = float("-inf")

        return logits

    def forward(self, **kwargs):
        
        x = self.w2v_encoder(**kwargs)
        return x



class HubertEncoder(FairseqEncoder):
    def __init__(self, cfg: HubertAsrConfig, tgt_dict=None):
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
        # w2v_args.task.labels = task.cfg.labels
        # w2v_args.task.label_dir = task.cfg.label_dir
        #w2v_args.task.label_dir = "/modelblob/users/v-sanych/data/librispeech/hubert_iter1_layer6_km_label/train_960/k500/"
        task = tasks.setup_task(w2v_args.task)
        if state is not None and "task_state" in state:
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

        self.predict_layers = eval(cfg.predict_layers)
        self.separate_ctc_layer = cfg.separate_ctc_layer

        self.w2v_model.remove_pretraining_modules()
        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if len(self.predict_layers) > 1:
            if self.separate_ctc_layer:
                self.proj = nn.Sequential(
                    *[Linear(d, len(tgt_dict)) for _ in range(len(self.predict_layers))])
            else:
                self.proj = nn.Sequential(
                    *[Linear(d, len(tgt_dict))] * len(self.predict_layers))
        else:
            self.proj = Linear(d, len(tgt_dict))
        """
        if hasattr(model, 'w2v_encoder'):
            if hasattr(model.w2v_encoder, 'proj'):
                self.proj = model.w2v_encoder.proj
        elif tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj = None
        """

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
            "features_only": True
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model(**w2v_args)
            x = res["x"]
            padding_mask = res["padding_mask"]

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)
        before_proj = x
        layer_results = res["layer_results"]

        if len(self.predict_layers) > 1:
            assert len(layer_results) == self.predict_layers[-1]
        if len(self.predict_layers) > 1:
            layer_results = [layer_x for i, (layer_x, _) in enumerate(layer_results) if (i+1) in self.predict_layers]

            encoder_out_list = []
            for layer_x, proj in zip(layer_results, self.proj):
                layer_x = self.final_dropout(layer_x)
                layer_x = proj(layer_x)
                encoder_out_list.append(layer_x)
            x = sum(encoder_out_list)
        else:
            # x = self.final_dropout(x)

            if isinstance(self.proj, torch.nn.Sequential):
                x = self.proj[-1](x)
            else:
                x = self.proj(x)

        return {
            "before_proj": before_proj,
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
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
