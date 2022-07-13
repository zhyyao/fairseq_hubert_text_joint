# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from omegaconf import II

import numpy as np
from fairseq import metrics, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
    Dictionary,
)
from typing import Any, Dict, List, Optional, Tuple, Union

from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
import fairseq

logger = logging.getLogger(__name__)
def load_pair_dataset(
    data_path,
    src_lang,
    trg_lang,
    src_dictionary,
    trg_dictionary,
    dataset_impl,
    shuffle,  
):
    src_data = os.path.join(data_path, src_lang+".txt")
    trg_data = os.path.join(data_path, trg_lang+".txt")

    src_dataset = data_utils.load_indexed_dataset(
        src_data, src_dictionary, dataset_impl
    )
    trg_dataset = data_utils.load_indexed_dataset(
        trg_data, trg_dictionary, dataset_impl
    )
    logger.info(
        "loaded src data{}, target data{}".format(
            src_data, trg_data
        )
    )
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dictionary,
        trg_dataset,
        trg_dataset.sizes,
        trg_dictionary,
        left_pad_source=False,
        left_pad_target=False,
        shuffle=shuffle,
    )


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False,
        )

@dataclass
class CorrectionConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    train_subset: str = II("dataset.train_subset")
    valid_subset: str = II("dataset.valid_subset")
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    label: Optional[str] = field(
        default="ltr",
        metadata={
            "help": "label of the text data",

        },
    )
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    pretrained_model: Optional[str] = field(
        default="/datablob/users/v-zhuoyao/model/librispeech_lm_bert_change_mask",
        metadata={
            "help": "path to pretrained mlm checkpoint",
        },
    )
    tokens_per_sample: int = field(
        default=250000,
        metadata={
            "help" : "batch tokens per sample"
        }
    )

@register_task("correction", dataclass=CorrectionConfig)
class CorrectionTask(FairseqTask):
    """
    Correction for ASR model output
    """

    cfg: CorrectionConfig

    def __init__(self,cfg,dictionary):
        import copy
        super().__init__(cfg)
        self.mask_idx = dictionary.add_symbol("<mask>")
        self.src_dictionary = dictionary
        self.trg_dictionary = copy.deepcopy(dictionary)
        self.trg_dictionary.add_symbol("<del>")
        self.dictionary = self.trg_dictionary

    @property
    def target_dictionary(self):
        return self.trg_dictionary
    @property
    def source_dictionary(self):
        return self.src_dictionary
    @classmethod
    def setup_task(cls, cfg: CorrectionConfig, **kwargs):
        """Setup the task 

        """
        paths = utils.split_paths(cfg.data)
        dictionary = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.label))
        )

        logger.info("[{}] dictionary: {} types".format(cfg.label, len(dictionary)))
        return cls(cfg, dictionary)

    def load_dataset(self, split, epoch=1, **kwargs):
        """Load a given dataset split.
        """
        path = self.cfg.data
        src = split+"."+self.cfg.source_lang
        trg = split+"."+self.cfg.target_lang

        self.datasets[split] = load_pair_dataset(
            path,
            src,
            trg,
            self.src_dictionary,
            self.trg_dictionary,
            dataset_impl=self.cfg.dataset_impl,
            shuffle=(split != "test")
        )
    
    def build_model(self, cfg: FairseqDataclass):
        """
        Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
        task.

        Args:
            cfg (FairseqDataclass): configuration object

        Returns:
            a :class:`~fairseq.models.BaseFairseqModel` instance
        """
        from fairseq import models, quantization_utils

        model = models.build_model(cfg, self)
        model = quantization_utils.quantize_model_scalar(model, cfg)
        loaded = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [cfg.pretrained_model+"/checkpoint_best.pt"]
            )
        ([roberta_enc], _cfg, _task) = loaded
        model.encoder.sentence_encoder = roberta_enc.encoder.sentence_encoder
        # model = model.from_pretrained(cfg.pretrained_model, checkpoint_file='checkpoint_best.pt')
        return model



