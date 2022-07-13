# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
from random import sample
import sys
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.audio.audio_utils import get_fbank
from tqdm import tqdm
import time
from omegaconf import II


logger = logging.getLogger(__name__)

def load_paired_data(manifest_path, max_keep, min_keep):
    n_long, n_short = 0,0
    data_dict, inds, sizes = [], [], []
    with open(manifest_path) as f:
        for ind, line in enumerate(f):
            items = line.strip().split(":")
            if len(items) ==6:

                sz = int(items[5])
                if min_keep is not None and sz < min_keep:
                    n_short += 1
                elif max_keep is not None and sz > max_keep:
                    n_long += 1
                else:
                    data_dict.append(
                        {
                            "id": items[1].split(" ")[0],
                            "path": items[2].split(" ")[0],
                            "phoneme": " ".join(items[3].split(" ")[0:-1]),
                            "word": " ".join(items[4].split(" ")[0:-1]),
                        }
                    )
                    inds.append(ind)
                    sizes.append(sz)
            elif len(items) == 5:
                sz = int(items[4])
                if min_keep is not None and sz < min_keep:
                    n_short += 1
                elif max_keep is not None and sz > max_keep:
                    n_long += 1
                else:
                    data_dict.append(
                        {
                            "id": items[1].split(" ")[0],
                            "path": items[2].split(" ")[0],
                            "word": " ".join(items[3].split(" ")[0:-1]),
                        }
                    )
                    inds.append(ind)
                    sizes.append(sz)
    logger.info(
        (
            f"load paired data"
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(data_dict)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return data_dict, inds, sizes

    

def load_text_only_data(text_only_data_set_path, phone_data_set_path, max_text, min_text, store_text):
    sizes = []
    s = open("per_sen_rep_store.txt")
    for  ind, sz in enumerate(s):
        sz=int(sz) 
        if max_text is not None and sz > max_text:
            sz = max_text
        elif min_text is not None and sz < min_text:
            sz = max_text
        sizes.append(sz)
    s.close()
    return sizes

def load_label_offset(label_path, inds):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets

class AudioDataset(FairseqDataset):
    def __init__(
        self,
        audio_path: str,
        sample_rate: float,
        max_keep_sample_size: int = None,
        min_keep_sample_size: int = None,
        label_processors: Optional[List[Any]] = None,
        pad_list: List[str] = None,
        eos_list: List[str] = None,
        shuffle: bool = True,
        pad_audio: bool = True,
        normalize: bool = False,
        fbank_bins: int = 80,
        max_sample_size: int=100000000,
    ):
        self.audio_data_dict, self.audio_inds, self.audio_sizes = load_paired_data(
            audio_path, max_keep_sample_size, min_keep_sample_size
        )

        self.sample_rate = sample_rate
        self.shuffle = shuffle

        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.fbank_bins = fbank_bins
        self.max_sample_size = max_sample_size
        self.normalize = normalize
        self.dataset = self
        self.pad_audio = pad_audio

    def __getitem__(self, index):
        # print("get wav")
        wav = self.get_audio(index)
        phoneme_token,bpe_token = self.get_label(index)
        if phoneme_token is not None:
            '''
                notice!!!
                phoneme > 10 is because of the 0-10 in the dictionary of phoneme is <eps>, SIL, SPN 
            '''
            phoneme_token_no_rep = torch.from_numpy(np.array( [ int(phoneme_token[i]) for i in range(1,len(phoneme_token)) if phoneme_token[i] > 10 and (i==1 or phoneme_token[i]!=phoneme_token[i-1]) ] ))
        else:
            phoneme_token_no_rep = None
        return {"id": index, "source": wav, "phoneme": phoneme_token, "bpe":bpe_token, "phoneme_target": phoneme_token_no_rep}

    def __len__(self):
        return len(self.sizes)

    @property
    def sizes(self):
        return self.audio_sizes

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def get_audio(self, index):
        import soundfile as sf

        wav_path = self.audio_data_dict[index]["path"]
        wav, cur_sample_rate = sf.read(wav_path)
        wav = torch.from_numpy(wav).float()
        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav

    def get_label(self, index):
        data = self.audio_data_dict[index]
        phoneme_token = None
        if "phoneme" in data.keys():
            phoneme_token = self.label_processors[1](data["phoneme"])
        bpe_token = self.label_processors[0](bpe_token)
        return phoneme_token, bpe_token

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)

        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios, audio_size
        )
        if samples[0]["phoneme"] is not  None:
            phoneme_input = [s["phoneme"] for s in samples]
            phoneme_target = [s["phoneme_target"] for s in samples] 
            phoneme_mask = self.phoneme_padding_mask(phoneme_input)
        else:
            phoneme_input = None
            phoneme_target = None
            phoneme_mask = None
        bpe_target = [s["bpe"] for s in samples]

        prev_phoneme,phoneme_t,phoneme_length=None,None,None
        if phoneme_input is not None:
            prev_phoneme, _, _ = self.collater_seq_label(
                phoneme_input, self.pad_list[0]
            )

        bpe_target, bpe_length, _ = self.collater_seq_label(
            bpe_target, self.pad_list[1]
        )
        
        if phoneme_target is not None:
            phoneme_t, phoneme_length, _ = self.collater_seq_label(
                phoneme_target, self.pad_list[0]
            )

        net_input = {
            "audio_source": collated_audios, 
            "padding_mask": padding_mask, 
            "prev_phoneme": prev_phoneme, 
            "phoneme_padding_mask": phoneme_mask,
            "mode": "speech",
            "lengths": ((torch.from_numpy(np.array(audio_sizes))- (400-320)) / 320).int()
        }
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }
        batch["phoneme_length"] = phoneme_length
        batch["phoneme_target"] = phoneme_t
        batch["bpe_length"] = bpe_length
        batch["bpe_target"] = bpe_target
        return batch

    def phoneme_padding_mask(self, phoneme_target):
        phoneme_sizes = [ len(s) for s in phoneme_target]
        max_size = max(phoneme_sizes)
        batch_size = len(phoneme_target)
        padd_mask = torch.zeros((batch_size, max_size)).bool()
        for  i, phoneme in enumerate(phoneme_target):
            diff =  max_size - len(phoneme) 
            if diff == 0:
                continue
            elif diff < 0:
                padd_mask[i, diff:] = True
        return padd_mask

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav[start:end], start

    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios]
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat(
                    [audio, audio.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )

        return collated_audios, padding_mask, audio_starts

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        
        return targets, lengths, ntokens

    
    def collater_label(self, phoneme_input, bpe_target, phoneme_target):
        targets=[None,None,None]
        lengths=[None,None,None]
        ntokens=[None,None,None]
        if phoneme_input is not None:
            targets[0], lengths[0], ntokens[0] = self.collater_seq_label(
                phoneme_input, self.pad_list[0]
            )
        targets[1], lengths[1], ntokens[1] = self.collater_seq_label(
            bpe_target, self.pad_list[1]
        )
        if phoneme_target is not None:
            targets[2], lengths[2], ntokens[2] = self.collater_seq_label(
                phoneme_target, self.pad_list[0]
            )

        return targets, lengths, ntokens

    def size(self, index):
        return self.sizes[index]

    def num_tokens(self, index: int):
        return self.size(index)

class TextDataset(FairseqDataset):
    def __init__(
        self,
        data_file_path: str,
        phone_data_file_path: str,
        accume_path: str,
        target_dictionary,
        phoneme_dictionary,
        seed,
        max_text_num:int = None,
        min_text_num:int = None,
        data_process:Optional[List[Any]] = None,
        shuffle: bool = True,
        pad_list: List[str] = None,
        eos_list: List[str] = None,
        store_text: bool = False,
        mask_prob: float = 0.15,
        dataset_impl: str = None,
        MASK: int = 0,
        PAD: int = 1,
    ):
        self.seed = seed
        self.target_dictionary = target_dictionary
        self.phoneme_dictionary = phoneme_dictionary
        self.rep_dict = self.load_accum_stat(accume_path)
        self.max_keep_sample_size = max_text_num
        self.min_keep_sample_size = min_text_num
        self.store_text = store_text
        self.text_sizes = load_text_only_data(
                data_file_path, phone_data_file_path, max_text_num, min_text_num,store_text
            ) 

        self.word = data_utils.load_indexed_dataset(
            data_file_path,
            self.target_dictionary,
        )

        self.phone = data_utils.load_indexed_dataset(
            phone_data_file_path,
            self.phoneme_dictionary,
        )

        self.shuffle = shuffle
        self.pad_list = pad_list
        self.dataset = self
        
        self.data_process = data_process
        self.eos_list = eos_list

        self.MASK = MASK
        self.PAD = PAD
        self._mask_prob = mask_prob

        self.epoch = 0
        self.last_ind = 0
        
        #self.word_f = open(self.word)
        #self.phone_f = open(self.phone)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @property
    def sizes(self):
        return self.text_sizes

    def avoid_zero(self, accum_stat,key):
        prefix = key.split("_")[0]
        if accum_stat[prefix+"_B"] + accum_stat[prefix+"_I"] + accum_stat[prefix+"_E"] + accum_stat[prefix+"_S"] ==0:
            accum_stat[prefix+"_B"] =1 
            accum_stat[prefix+"_I"] =1
            accum_stat[prefix+"_E"] =0
            accum_stat[prefix+"_S"] =0
        
    def load_accum_stat(self, accum_path):
        accum_stat = {}
        rep_dict = {}
        store = []
        with open(accum_path) as f:
            for  line in f.readlines():
                item = line.strip().split()
                accum_stat[item[0]]=int(item[1])
                store.append(int(item[1]))
            # min = np.min(store)
            # max = np.max(store)
            # scale = 8
            # for key in accum_stat.keys():
            #     accum_stat[key] = int(((accum_stat[key] -min) / (max-min)) * scale)
            for key in accum_stat.keys():
                self.avoid_zero(accum_stat, key)
        for  key in accum_stat.keys():
            phoneme = key.split("_")[0]
            if phoneme not in rep_dict.keys():
                rep_dict[self.phoneme_dictionary.index(phoneme)] = ( accum_stat[phoneme+"_B"] + \
                                accum_stat[phoneme+"_I"] + \
                                accum_stat[phoneme+"_E"] + \
                                accum_stat[phoneme+"_S"] )
        for p in self.phoneme_dictionary.symbols:
            if self.phoneme_dictionary.index(p) not in rep_dict.keys():
                rep_dict[self.phoneme_dictionary.index(p)] = 10
                logger.info("missing phone: "+p)

                
        return rep_dict

        
    def __getitem__(self, index):
        # print("get text")
        phoneme_token_target,bpe_token, phoneme_token = self.get_labels(index)
        return {"id": index,  "phoneme": phoneme_token, "bpe":bpe_token, "phoneme_target": phoneme_token_target}
        
    def get_labels(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            words = self.word[index][:-1]
            phoneme_seq = self.phone[index][:-1]
            bpe_token = words
            phoneme_list = []
            phoneme_src_list = []
            # logging.info("index: "+str(index)+" "+self.phoneme_dictionary.string(phoneme_seq)+str(self.phone[index][:-1]))
            for ind in range(phoneme_seq.shape[0]):
                if(np.random.rand() < self._mask_prob):
                    p = phoneme_seq[ind].item() 
                    phoneme_list.extend([p for i in range(self.rep_dict[p])])
                    phoneme_src_list.extend([self.MASK for i in range(self.rep_dict[ p ]) ])
                else:
                    p = phoneme_seq[ind].item()
                    phoneme_src_list.extend([ p for i in range(self.rep_dict[ p ]) ] )
                    phoneme_list.extend( [ self.PAD for i in range(self.rep_dict[ p ]) ] ) 
            if self.max_keep_sample_size is not None and len(phoneme_src_list) > self.max_keep_sample_size:
                words = self.word[self.last_ind][:-1]
                phoneme_seq = self.phone[self.last_ind][:-1]
                phoneme_list = []
                phoneme_src_list = []
                for ind in range(phoneme_seq.shape[0]):
                    if(np.random.rand() < self._mask_prob):
                        p = phoneme_seq[ind].item()
                        phoneme_list.extend([p for i in range(self.rep_dict[ p ])])
                        phoneme_src_list.extend([self.MASK for i in range(self.rep_dict[ p ]) ])
                    else:
                        p = phoneme_seq[ind].item()
                        phoneme_src_list.extend([ p for i in range(self.rep_dict[ p ]) ] )
                        phoneme_list.extend( [ self.PAD for i in range(self.rep_dict[ p ]) ] ) 
            elif self.min_keep_sample_size is not None and len(phoneme_src_list) < self.min_keep_sample_size:
                words = self.word[self.last_ind][:-1]
                phoneme_seq = self.phone[self.last_ind][:-1]
                phoneme_list = []
                phoneme_src_list = []
                for ind in range(phoneme_seq.shape[0]):
                    if(np.random.rand() < self._mask_prob):
                        p = phoneme_seq[ind].item()
                        phoneme_list.extend([p for i in range(self.rep_dict[ p ])])
                        phoneme_src_list.extend([self.MASK for i in range(self.rep_dict[ p ]) ])
                    else:
                        p = phoneme_seq[ind].item()
                        phoneme_src_list.extend([ p for i in range(self.rep_dict[ p ]) ] )
                        phoneme_list.extend( [ self.PAD for i in range(self.rep_dict[ p ]) ] ) 
            else:
                self.last_ind = index
        phoneme_list = torch.tensor(phoneme_list)
        phoneme_src_list = torch.tensor(phoneme_src_list)
        return phoneme_list, bpe_token, phoneme_src_list

    def size(self, index):
        return self.sizes[index]

    def num_tokens(self, index: int):
        return self.sizes[index]

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def collater(self, samples):
        phoneme_input = [s["phoneme"] for s in samples]
        lengths = [len(s) for s in phoneme_input]
        bpe_output = [s["bpe"] for s in samples ]
        phoneme_target = [s["phoneme_target"] for s in samples]
        phoneme_mask = self.phoneme_padding_mask(phoneme_input)
        phoneme_input, phoneme_lengths, phoneme_ntokens = self.collater_seq_label(
            phoneme_input, self.pad_list[1], self.eos_list[1]
        )
        bpe_output, bpe_lengths, bpe_ntokens = self.collater_seq_label(
            bpe_output, self.pad_list[0], self.eos_list[0]
        )
        phoneme_target, phoneme_lengths, phoneme_ntokens = self.collater_seq_label(
            phoneme_target, self.pad_list[1], self.eos_list[1]
        )
        net_input = {
            "source": None, 
            "padding_mask": None, 
            "phf_input": phoneme_input, 
            "phf_padding_mask": phoneme_mask,
            "mode":"text_only",
            "lengths":phoneme_lengths
        }
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }
        batch["bpe_ntokens"] = bpe_ntokens
        batch["bpe_target"] = bpe_output
        batch["bpe_length"] = bpe_lengths
        batch["phoneme_ntokens"] = phoneme_ntokens
        batch["phoneme_target"] = phoneme_target
        batch["phoneme_length"] = phoneme_lengths
        return batch

    def collater_seq_label(self, targets, pad, eos):
        lengths = torch.LongTensor([len(t) for t in targets])
        
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        
        return targets, lengths, ntokens

    def __len__(self):
        return len(self.sizes)

    def phoneme_padding_mask(self, phoneme_target):
        phoneme_sizes = [ len(s) for s in phoneme_target]
        max_size = max(phoneme_sizes)
        batch_size = len(phoneme_target)
        padd_mask = torch.zeros((batch_size, max_size)).bool()
        for  i, phoneme in enumerate(phoneme_target):
            diff = len(phoneme) - max_size
            if diff == 0:
                continue
            else:
                padd_mask[i,diff:]=True
        return padd_mask



