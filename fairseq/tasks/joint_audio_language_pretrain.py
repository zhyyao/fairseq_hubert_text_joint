# v-zhuoyao@microsoft.com
# 2021 / 10 / 27
# Define the task of optimizing alignment of speech and language latent spaces for e2e speech
# recognition and understanding
# arxiv:
import logging
import os
from random import shuffle
import sys
from typing import Dict, List, MutableMapping, Optional, Tuple, Union
from fairseq.dataclass import configs
from fairseq.utils import index_put

import numpy as np
import sentencepiece as spm

from dataclasses import dataclass, field
from fairseq.data import Dictionary
from fairseq.data.audio.audio_text_dataset import AudioDataset, TextDataset
from fairseq.data.audio.audio_dataset import AudioDataset
from fairseq.data.audio.multitask_dataset import MultitaskDataset
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING, II

from fairseq.data import (
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
    ModeDataset
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset

logger = logging.getLogger(__name__)

@dataclass
class JointAudioLanguagePretrainConfig(FairseqDataclass):
    data: str = field(
        default=MISSING, metadata={"help" : "path to data directory"}
    )
    fine_tuning: bool = field(
        default=False, metadata={"help": "set to true if fine-tuning Hubert"}
    )
    text_data: str = field(
        default=MISSING, metadata={"help": "path to text only data directory"}
    )
    phone_text_data:str = field(
        default=MISSING, metadata={"help": "path to phone text only data directory"}
    )
    multi_task_subset: List[str] = field(
        default_factory=lambda: ["librilight_vox_giga", "train_clean_100"],
        metadata={
            "help": (
                "multi task audio subset"
            )
        }
    )
    valid_subset: str= field(
        default=MISSING, metadata={"help": "valid data file"}
    )

    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={
            "help": "if set, normalizes input to have 0 mean and unit variance"
        },
    )
    max_keep_size: Optional[int] = field(
        default=1000000000,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    text_max_sample: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size for text dataset"},
    )
    text_min_sample: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size for text dataset"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    shuffle: bool=field(
        default=True,
        metadata={"help":"shuffle the dataset or not"}
    )
    fbank_bin:bool = field(
        default=80,
        metadata={"help": "fbank bins of the model"}
    )
    accum_path: str = field(
        default=MISSING,
        metadata={"help": "accumulate file of phoneme"}
    )
    sample_ratio: str = field(
        default="1:1",
        metadata={"help": "token num speech:text"}
    )
    batch_ratio: str = field(
        default="1:0.5",
        metadata={"help": "batch size speech: text"}
    )
    aband_text_encoder: bool=field(
        default=False,
        metadata={"help": "abandond text encoder"}
    )
    aband_shared_encoder: bool=field(
        default=False,
        metadata={"help": "abandond shared encoder"}
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys "
            "as AddTargetDataset"
        },
    )
    text_encoder_mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
            "(normalized by length)"
        },
    )
    text_encoder_mask_length: int = field(
        default = 10,
        metadata={
            "help": "length of text mask"
        }
    )
    text_encoder_mask_stdev: float = field(
        default = 1,
        metadata = {
            "help": "stdev of mask length of text data"
        }
    )

    seed: int = II("common.seed")
    only_speech: Optional[bool] = field(
        default=False,
        metadata={
            "help": "only use speech data"
        }
    )
    

@register_task("joint_audio_language_pretrain", dataclass=JointAudioLanguagePretrainConfig)
class JointAudioLanguagePretrainTask(FairseqTask):

    cfg: JointAudioLanguagePretrainConfig
    

    def __init__(
        self,
        cfg: JointAudioLanguagePretrainConfig
    ) -> None:
        super().__init__(cfg)
        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"JointAudioLanguagePretrain Config {cfg}")

        self.cfg = cfg

        self.blank_symbol = "<s>"
        self.state.add_factory("target_dictionary", self.load_target_dictionary)
        self.state.add_factory("phoneme_dictionary", self.load_phoneme_dictionaries)

    
    @classmethod
    def setup_task(
        cls, cfg: JointAudioLanguagePretrainConfig, **kwargs
    ) -> "OptimizingAlignmentTask":
        return cls(cfg)

    def load_target_dictionary(self):
        if self.cfg.labels:
            if self.cfg.dict_path is not None:
            #if "dict_path" in self.cfg:
                target_dictionary = Dictionary(self.cfg.dict_path, self.cfg.dict_model)
            else:
                dict_path = os.path.join(self.cfg.data, f"letter.json")
                target_dictionary = Dictionary(dict_path)
            return target_dictionary
        return None

    def load_phoneme_dictionaries(self):
        data_dir = os.path.dirname(self.cfg.phone_text_data)
        dic = Dictionary.load(f"{data_dir}/dict.txt")
        self.MASK = dic.add_symbol("<mask>")
        self.PAD = dic.pad()
        return dic

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    @property
    def phoneme_dictionary(self) -> Optional[Dictionary]:
        return self.state.phoneme_dictionary

    def load_dataset(self, split:str, **kwargs) ->None:
        sample_ratio = [ float(i) for i in self.cfg.sample_ratio.split(":")]
        batch_ratio = [ float(i) for i in self.cfg.batch_ratio.split(":")]

        datasets=[]
        if split.startswith("multi_task") :
            multi_task_list = self.cfg.multi_task_subset
            for index, subset in  enumerate(multi_task_list):
                manifest = f"{self.cfg.data}/{subset}.tsv"
                    
                audio_dataset = AudioDataset(
                    manifest_path=manifest,
                    sample_rate=self.cfg.sample_rate,
                    max_keep_sample_size=None,
                    min_keep_sample_size=self.cfg.min_sample_size,
                    max_sample_size=self.cfg.max_sample_size,
                    shuffle=self.cfg.shuffle,
                    normalize=self.cfg.normalize,
                    pad_audio=self.cfg.pad_audio,
                    random_crop=self.cfg.random_crop,
                )
                datasets.append(audio_dataset)

            if not self.cfg.only_speech:
                path = self.cfg.phone_text_data
                dataset = data_utils.load_indexed_dataset(
                    path,
                    self.phoneme_dictionary,
                )
                if dataset is None:
                    raise FileNotFoundError(
                        "Dataset not found: {} ({})".format(split, split_path)
                    )
                logging.info("split")

                dataset = maybe_shorten_dataset(
                    dataset,
                    split,
                    "",
                    "none",
                    self.cfg.text_max_sample,
                    self.cfg.seed,
                )

                logging.info("token block")
                # create continuous blocks of tokens
                dataset = TokenBlockDataset(
                    dataset,
                    dataset.sizes,
                    self.cfg.text_max_sample - 1,  # one less for <s>
                    pad=self.phoneme_dictionary.pad(),
                    eos=self.phoneme_dictionary.eos(),
                    break_mode="complete",
                )
                logger.info("loaded {} blocks from: {}".format(len(dataset), path))

                # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
                dataset = PrependTokenDataset(dataset, self.phoneme_dictionary.bos())

                # create masked input and targets
                src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
                    dataset,
                    self.phoneme_dictionary,
                    pad_idx=self.phoneme_dictionary.pad(),
                    mask_idx=self.MASK,
                    seed=self.cfg.seed,
                    mask_prob=self.cfg.text_encoder_mask_prob,
                    mask_multiple_length = self.cfg.text_encoder_mask_length,
                    mask_stdev = self.cfg.text_encoder_mask_stdev
                )

                with data_utils.numpy_seed(self.cfg.seed):
                    shuffle = np.random.permutation(len(src_dataset))

                text_dataset = SortDataset(
                    NestedDictionaryDataset(
                        {
                            "id": IdDataset(),
                            "net_input": {
                                "source": RightPadDataset(
                                    src_dataset,
                                    pad_idx=self.phoneme_dictionary.pad(),
                                ),
                                "lengths": NumelDataset(src_dataset, reduce=False),
                                "mode": ModeDataset(src_dataset,"text_only"),
                                "phoneme_target": RightPadDataset(
                                    dataset,
                                    pad_idx=self.phoneme_dictionary.pad(),
                                ),
                            },
                            "nsentences": NumSamplesDataset(),
                            "phoneme_ntokens": NumelDataset(src_dataset, reduce=True),
                            "phoneme_length": NumelDataset(src_dataset, reduce=False),
                        },
                        sizes=[src_dataset.sizes],
                    ),
                    sort_order=[
                        shuffle,
                        src_dataset.sizes,
                    ],
                )
                
                datasets.append(text_dataset)
                
            self.datasets[split] = MultitaskDataset(
                datasets=datasets,
                sample_ratios=sample_ratio,
                batch_ratio=batch_ratio
            )
        else:
            logger.info(f"load {split} dataset")
            manifest = f"{self.cfg.data}/{split}.tsv"
            
            audio_dataset = AudioDataset(
                manifest_path=manifest,
                sample_rate=self.cfg.sample_rate,
                max_keep_sample_size=None,
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_sample_size,
                shuffle=False,
                normalize=self.cfg.normalize,
                pad_audio=self.cfg.pad_audio,
                random_crop=self.cfg.random_crop,
            )
            
            self.datasets[split] = audio_dataset
            logger.info(str(self.datasets[split]))
    
    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self, indices: np.array, *args, **kwargs
    ) -> np.array:
        return indices
