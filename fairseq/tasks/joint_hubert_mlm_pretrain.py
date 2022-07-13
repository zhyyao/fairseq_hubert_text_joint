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
from fairseq.data.audio.hubert_dataset import HubertDataset
from fairseq.data.audio.multitask_dataset import MultitaskDataset
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING, II

from fairseq.data import (
    Dictionary,
    IdDataset,
    MaskAndRepDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    PostpendTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
    ModeDataset,
    DownSampleDataset
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset

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

class BpeEncoder(object):
    def __init__(self, model: str, dictionary:Dictionary) -> None:
        self.sentencepiece_processor = SentencepiecesTokenizer(model)
        self.dictionary = dictionary
    def __call__(self, label:str )-> List[str]:
        bpe_seq = self.sentencepiece_processor(label)
        bpe_seq = " ".join([ i for i in bpe_seq if i != "'" and i != " "])
        return self.dictionary.encode_line(
            bpe_seq, append_eos=False, add_if_not_exist=False,
        )

class CharEncoder(object):
    def __init__(self, dictionary:Dictionary) -> None:
        self.dictionary = dictionary
    def __call__(self, label:str )-> List[str]:
        char_seq = label.upper().replace(" ","|").replace(""," ")
        return self.dictionary.encode_line(
            char_seq, append_eos=False, add_if_not_exist=False,
        )

@dataclass
class JointSpeechTextPretrainConfig(FairseqDataclass):
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
    rep_file: str = field(
        default="", metadata={"help": "repeat num file of phf, None for no repeat"}
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
    label_dir: str = field(
        default=MISSING, metadata={"help": "path to label"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    multi_task_labels: str = field(
        default=' ltr |',
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_rate: int = field(
        default=-1,
        metadata={"help": "label frame rate. -1 for sequence label"},
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
        default = 1,
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
    world_size: int = II("distributed_training.distributed_world_size")
    only_speech: Optional[bool] = field(
        default=False,
        metadata={
            "help": "only use speech data"
        }
    )
    
    

@register_task("joint_hubert_mlm_pretrain", dataclass=JointSpeechTextPretrainConfig)
class JointHubertMlmPretrainTask(FairseqTask):

    cfg: JointSpeechTextPretrainConfig
    

    def __init__(
        self,
        cfg: JointSpeechTextPretrainConfig
    ) -> None:
        super().__init__(cfg)
        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"JointSpeechTextPretrainTask Config {cfg}")

        self.cfg = cfg
        
        self.state.add_factory("dictionaries", self.load_dictionaries)

        self._source_dictionary = None
        self.blank_symbol = "<blank>"
        self.has_text_encoder = not self.cfg.aband_text_encoder
        self.has_shared_encoder = not self.cfg.aband_shared_encoder
        self.mask_prob = self.cfg.text_encoder_mask_prob
        if self.cfg.rep_file == "":
            self.rep_proc = None
        else:
            self.rep_proc = self.cfg.rep_file


    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return self._source_dictionary

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.state.dictionaries[1]

    @property
    def phoneme_dictionary(self) -> Optional[Dictionary]:
        return self.state.dictionaries[2]

    @property
    def dictionaries(self) -> List[Dictionary]:
        return self.state.dictionaries
    
    @classmethod
    def setup_task(
        cls, cfg: JointSpeechTextPretrainConfig, **kwargs
    ) -> "OptimizingAlignmentTask":
        return cls(cfg)

    def load_dictionaries(self):
        
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        dictionaries = []
        for label in self.cfg.labels:
            if label == "km":
                dictionaries.append(Dictionary.load(f"{self.cfg.label_dir}/dict.km.txt"))
            elif label == "phf":
                data_dir = os.path.dirname(self.cfg.phone_text_data)
            
                dic = Dictionary.load(f"{data_dir}/dict.txt")
                self.MASK = dic.add_symbol("<mask>")
                dictionaries.append(dic)
                self.PAD = dic.pad()
            else:
                data_dir = os.path.dirname(self.cfg.text_data)
                dic = Dictionary.load(f"{data_dir}/dict.txt")
                dic.add_symbol("<blank>")
                dictionaries.append(dic)
        
        self.PHF_INDEX = self.cfg.labels.index("phf")
            
        return dictionaries

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    def decode_to_list(self,ss):
        r = []
        s_list = ss.split("|")
        for s_l in s_list:
            r.append(s_l.split())
        return r
        


    def load_dataset(self, split:str, **kwargs) ->None:
        sample_ratio = [ float(i) for i in self.cfg.sample_ratio.split(":")]
        batch_ratio = [ float(i) for i in self.cfg.batch_ratio.split(":")]

        datasets=[]
        if split.startswith("multi_task") :
            multi_task_list = self.cfg.multi_task_subset
            for index, subset in  enumerate(multi_task_list):
                manifest = f"{self.cfg.data}/{subset}.tsv"


                labels = self.decode_to_list(self.cfg.multi_task_labels)
                label_index = [ self.cfg.labels.index(label) for label in labels[index]]
                dicts = [ dic for ind, dic in enumerate(self.dictionaries) if ind in label_index ]
                
                
                pad_list = [dict.pad() for dict in dicts]
                eos_list = [dict.eos() for dict in dicts]
                bos_list = [dict.eos() for dict in dicts]

                procs = [LabelEncoder(dict) for dict in dicts]

                paths =[]
                for  l in labels[index]:
                    if l == "km":
                        paths.append(f"{self.get_label_dir()}/{subset}.{l}")
                    else:
                        paths.append(f"{self.cfg.data}/{subset}.{l}")
                PHF_INDEX = -1 if "km" in labels[index] and len(labels[index])==1 else 1
                
                if "phf" in labels[index]:
                    audio_dataset = HubertDataset(
                        manifest_path=manifest,
                        sample_rate=self.cfg.sample_rate,
                        label_paths=paths,
                        label_rates=[self.cfg.label_rate for p in paths],
                        label_processors=procs,
                        pad_list=pad_list,
                        eos_list=eos_list,
                        phf_index=PHF_INDEX,
                        max_keep_sample_size=self.cfg.max_sample_size,
                        min_keep_sample_size=self.cfg.min_sample_size,
                        max_sample_size= None,
                        shuffle=self.cfg.shuffle,
                        normalize=self.cfg.normalize,
                        pad_audio=True,
                        store_labels=False,
                        random_crop=self.cfg.random_crop,
                        single_target=self.cfg.single_target,
                    )
                    datasets.append(audio_dataset)
                else:
                    audio_dataset = HubertDataset(
                        manifest_path=manifest,
                        sample_rate=self.cfg.sample_rate,
                        label_paths=paths,
                        label_rates=self.cfg.label_rate,
                        label_processors=procs,
                        pad_list=pad_list,
                        eos_list=eos_list,
                        phf_index=PHF_INDEX,
                        max_keep_sample_size=None,
                        min_keep_sample_size=self.cfg.min_sample_size,
                        max_sample_size=self.cfg.max_sample_size,
                        shuffle=self.cfg.shuffle,
                        normalize=self.cfg.normalize,
                        pad_audio=self.cfg.pad_audio,
                        store_labels=False,
                        random_crop=self.cfg.random_crop,
                        single_target=self.cfg.single_target,
                    )
                    datasets.append(audio_dataset)
        
            
            if not self.cfg.only_speech:
                path = self.cfg.phone_text_data
                dataset = data_utils.load_indexed_dataset(
                    path,
                    self.phoneme_dictionary,
                )
                ltr_path = self.cfg.text_data
                ltr_dataset = data_utils.load_indexed_dataset(
                    ltr_path,
                    self.target_dictionary,
                )
                if dataset is None:
                    raise FileNotFoundError(
                        "Dataset not found: {} ({})".format(split, split_path)
                    )
                logging.info("split")

                # dataset = maybe_shorten_dataset(
                #     dataset,
                #     split,
                #     "",
                #     "random_crop",
                #     self.cfg.text_max_sample,
                #     self.cfg.seed,
                # )

                # logging.info("token block")
                # # create continuous blocks of tokens
                # dataset = TokenBlockDataset(
                #     dataset,
                #     dataset.sizes,
                #     self.cfg.max_sample_size * batch_ratio[1] - 1,  # one less for <s>
                #     pad=self.phoneme_dictionary.pad(),
                #     eos=self.phoneme_dictionary.eos(),
                #     break_mode="complete",
                # )
                # logger.info("loaded {} blocks from: {}".format(len(dataset), path))

                # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
                #dataset = PrependTokenDataset(dataset, self.phoneme_dictionary.bos())
                ltr_dataset = PostpendTokenDataset(ltr_dataset, self.target_dictionary.index("|"))

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
                                "phf_input": RightPadDataset(
                                    src_dataset,
                                    pad_idx=self.phoneme_dictionary.pad(),
                                ),
                                "lengths": NumelDataset(src_dataset, reduce=False),
                                "mode": ModeDataset(src_dataset,"text_only"),
                            },
                            "phoneme_target": RightPadDataset(
                                tgt_dataset,
                                pad_idx=self.phoneme_dictionary.pad(),
                            ),
                            "ltr_target": RightPadDataset(
                                ltr_dataset,
                                pad_idx=self.target_dictionary.pad(),
                            ),
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
                batch_ratio=batch_ratio,
                seed = self.cfg.seed,
                world_size = self.cfg.world_size
            )
        else:
            logger.info(f"load {split} dataset")
            manifest = f"{self.cfg.data}/{split}.tsv"
            labels = ["km"]
            label_index = [ self.cfg.labels.index(label) for label in labels]
            dicts = [ dic for ind, dic in enumerate(self.dictionaries) if ind in label_index ]
            
            pad_list = [dict.pad() for dict in dicts]
            eos_list = [dict.eos() for dict in dicts]
            bos_list = [dict.eos() for dict in dicts]

            procs = [LabelEncoder(dict) for dict in dicts]
            paths = []
            for  l in labels:
                    if l == "km":
                        paths.append(f"{self.get_label_dir()}/{split}.{l}")
                    else:
                        paths.append(f"{self.cfg.data}/{split}.{l}")

            audio_dataset = HubertDataset(
                manifest_path=manifest,
                sample_rate=self.cfg.sample_rate,
                label_paths=paths,
                label_rates=self.cfg.label_rate,
                label_processors=procs,
                pad_list=pad_list,
                eos_list=eos_list,
                phf_index=-1,
                max_keep_sample_size=self.cfg.max_sample_size,
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_sample_size,
                shuffle=self.cfg.shuffle,
                normalize=self.cfg.normalize,
                pad_audio=self.cfg.pad_audio,
                store_labels=False,
                random_crop=self.cfg.random_crop,
                single_target=self.cfg.single_target,
                is_valid=True
            )
            
            self.datasets[split] = audio_dataset
            logger.info(str(self.datasets[split]))
    
    def max_positions(self):
        return self.cfg.max_sample_size

