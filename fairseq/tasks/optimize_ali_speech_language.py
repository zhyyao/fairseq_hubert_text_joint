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
from fairseq.data.audio.multitask_dataset import MultitaskDataset
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False,
        )

class SentencepiecesTokenizer(object):
    def __init__(self, model: str):
        self.model = str(model)
        # NOTE(kamo):
        # Don't build SentencePieceProcessor in __init__()
        # because it's not picklable and it may cause following error,
        # "TypeError: can't pickle SwigPyObject objects",
        # when giving it as argument of "multiprocessing.Process()".
        self.sp = None

    def _build_sentence_piece_processor(self):
        # Build SentencePieceProcessor lazily.
        if self.sp is None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.model)

    def __call__(self, line) -> List[str]:
        self._build_sentence_piece_processor()
        return self.sp.EncodeAsPieces(line)


@dataclass
class OptimizingAlignmentConfig(FairseqDataclass):
    data: str = field(
        default=MISSING, metadata={"help" : "path to data directory"}
    )
    speech_data: str = field(
        default=MISSING, metadata={"help": "path to speech data directory"}
    )
    text_data: str = field(
        default=MISSING, metadata={"help": "path to text only data directory"}
    )
    phone_text_data:str = field(
        default=MISSING, metadata={"help": "path to phone text only data directory"}
    )
    valid_subset: str= field(
        default=MISSING, metadata={"help": "valid data file"}
    )
    label_dir: str = field(
        default=MISSING, metadata={"help": "path to label"}
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
    pad_audio: Optional[bool] = field(
        default=True,
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
    has_text_encoder: bool=field(
        default=True,
        metadata={"help": "use text encoder"}
    )
    
    


@register_task("optimizing_alignment_speech_language", dataclass=OptimizingAlignmentConfig)
class OptimizingAlignmentTask(FairseqTask):

    cfg: OptimizingAlignmentConfig

    def __init__(
        self,
        cfg: OptimizingAlignmentConfig
    ) -> None:
        super().__init__(cfg)
        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"HubertPretrainingTask Config {cfg}")

        self.cfg = cfg

        self.state.add_factory("dictionaries", self.load_dictionaries)

        self.blank_symbol = "<s>"

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.state.dictionaries[1]['bpe']

    @property
    def phoneme_dictionary(self) -> Optional[Dictionary]:
        return self.state.dictionaries[1]['phoneme']

    @property
    def dictionaries(self) -> List[Dictionary]:
        return self.state.dictionaries
    
    @classmethod
    def setup_task(
        cls, cfg: OptimizingAlignmentConfig, **kwargs
    ) -> "OptimizingAlignmentTask":
        return cls(cfg)

    def load_dictionaries(self):
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        dictionaries = [
	None,
	{
            "phoneme":Dictionary.load(f"{label_dir}/dict.phoneme.txt"),
            "bpe":Dictionary.load(f"{label_dir}/dict.bpe.txt") 
        }
	]
        self.MASK = dictionaries[1]["phoneme"].add_symbol("<mask>")
        return dictionaries

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    def load_dataset(self, split:str, **kwargs) ->None:
        bpe_model = f"{self.cfg.label_dir}/bpe_model/bpe.model"
        dicts = self.dictionaries
        
        pad_list = [self.phoneme_dictionary.pad(), self.target_dictionary.pad() ]
        eos_list = [self.phoneme_dictionary.eos(), self.target_dictionary.eos() ]
        bos_list = [self.phoneme_dictionary.bos(), self.target_dictionary.bos() ]

        procs = {
            "phoneme":LabelEncoder(dicts[1]["phoneme"]), 
            "bpe": LabelEncoder(dicts[1]["bpe"]),
            "word":SentencepiecesTokenizer(bpe_model) 
        }
        sample_ratio = [ float(i) for i in self.cfg.sample_ratio.split(":")]
        batch_ratio = [ float(i) for i in self.cfg.batch_ratio.split(":")]
        if split == "train":
            
            audio_dataset = AudioDataset(
                audio_path=self.cfg.speech_data,
                sample_rate=self.cfg.sample_rate,
                label_processors=procs,
                pad_list=pad_list,
                eos_list=eos_list,
                max_keep_sample_size=self.cfg.max_sample_size,
                min_keep_sample_size=self.cfg.min_sample_size,
                shuffle=self.cfg.shuffle,
                normalize=self.cfg.normalize,
                pad_audio=self.cfg.pad_audio,
                fbank_bins=self.cfg.fbank_bin,
                max_sample_size=self.cfg.max_keep_size,
            )
            if self.cfg.has_text_encoder:
                text_max_sample = None
                text_min_sample = None
                # 60 here means XXX_B/I/E/S * 3 = 4 * 3 = 12 phoneme_bies -> phoneme, phoneme -> word ,may be one word has avg 5 phoneme 
                if self.cfg.max_sample_size is not None:
                    text_max_sample = (self.cfg.max_sample_size - 80) // 320 
                if self.cfg.min_sample_size is not None:
                    text_min_sample = (self.cfg.min_sample_size - 80) // 320

                text_dataset = TextDataset(
                    data_file_path=self.cfg.text_data,
                    phone_data_file_path=self.cfg.phone_text_data,
                    max_text_num=text_max_sample,
                    min_text_num=text_min_sample,
                    data_process=procs,
                    shuffle=self.cfg.shuffle,
                    pad_list=pad_list,
                    accume_path=self.cfg.accum_path,
                    eos_list=eos_list,
                )
                
                self.datasets[split] = MultitaskDataset(
                    datasets=[audio_dataset, text_dataset],
                    sample_ratios=sample_ratio,
                    batch_ratio=batch_ratio
                )
            else:
                self.datasets[split] = audio_dataset
        else:
            audio_dataset = AudioDataset(
                audio_path=self.cfg.valid_subset,
                sample_rate=self.cfg.sample_rate,
                label_processors=procs,
                pad_list=pad_list,
                eos_list=eos_list,
                max_keep_sample_size=self.cfg.max_sample_size,
                min_keep_sample_size=self.cfg.min_sample_size,
                shuffle=self.cfg.shuffle,
                normalize=self.cfg.normalize,
                pad_audio=self.cfg.pad_audio,
                fbank_bins=self.cfg.fbank_bin,
                max_sample_size=self.cfg.max_keep_size,
            )
            self.datasets[split] = audio_dataset
