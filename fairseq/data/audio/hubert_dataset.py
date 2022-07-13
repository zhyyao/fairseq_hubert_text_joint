# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
import io
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)
from fairseq.data.audio.base_enh import *
import time
import soundfile as sf
logger = logging.getLogger(__name__)



def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )


class HubertDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        label_paths: List[str],
        label_rates: Union[List[float], float],  # -1 for sequence labels
        pad_list: List[str],
        eos_list: List[str],
        phf_index: int, 
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        store_labels: bool = True,
        random_crop: bool = False,
        single_target: bool = False,
        multitask: bool = False,
        is_valid: bool = False,
        noise_path: str =None,
        rir_path: str = None,
    ):
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.multitask = multitask
        self.epoch = 0
        self.audios = []

        self.chunk_names = []
        self.chunk_indices = []

        self.PHF_INDEX = phf_index
        self.is_valid = is_valid

        n_long, n_short = 0, 0
        names, inds, sizes = [], [], []
        bnds = []
        bnd_path = manifest_path.replace('tsv', 'bnd')
        if os.path.exists(bnd_path):
            with open(bnd_path) as f:
                bnds = f.readlines()
        new_bnds = []
        ind_final=0
        with open(manifest_path) as f:
            root = f.readline().strip()
            for ind, line in enumerate(f):
                items = line.strip().split("\t")
                sz = int(items[1])
                ind_final = ind
                if min_keep_sample_size is not None and sz < min_keep_sample_size:
                    n_short += 1
                elif max_keep_sample_size is not None and sz > max_keep_sample_size:
                    n_long += 1
                else:
                    fname = items[0].split(":")
                    if len(fname) > 1:
                        if len(self.chunk_names) == 0 or fname[0] != self.chunk_names[-1]:
                            self.chunk_names.append(fname[0])
                            self.chunk_indices.append(len(names))
                    names.append(items[0])
                    inds.append(ind)
                    sizes.append(sz)
                    if len(bnds) > 0:
                        new_bnds.append(list(map(int, bnds[ind].strip().split())))

        
        tot = ind_final + 1
        
        logger.info(
            (
                f"max_keep={max_keep_sample_size}, min_keep={min_keep_sample_size}, "
                f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
                f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
            )
        )
        self.audio_root = root
        self.audio_names = names
        self.sizes = sizes
        self.bnds = new_bnds
        self.add_noise = False
        if noise_path and rir_path is not None:
            self.add_noise = True
            noise_names, noise_inds, noise_sizes = [], [], []
            with open(noise_path) as f:
                root = f.readline().strip()
                for ind, line in enumerate(f):
                    items = line.strip().split("\t")
                    ind_final = ind
                    fname = items[0]
                    noise_names.append(items[0])
                    noise_inds.append(ind)
            self.noise_root = root
            self.noise_names = noise_names
            rir_names, rir_inds, rir_sizes = [], [], []
            with open(noise_path) as f:
                root = f.readline().strip()
                for ind, line in enumerate(f):
                    items = line.strip().split("\t")
                    ind_final = ind
                    fname = items[0]
                    rir_names.append(items[0])
                    rir_inds.append(ind)
            self.rir_root = root
            self.rir_names = rir_names

        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, int)
            else label_rates
        )
        self.store_labels = store_labels
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        
        assert (
            label_processors is None
            or len(label_processors) == self.num_labels
        )
        #for label_path, label_rate in zip(label_paths, self.label_rates):
        #    verify_label_lengths(
        #        self.sizes, sample_rate, label_path, label_rate, inds, tot
        #    )

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}"
        )

        # for i in range(len(self.audio_names)):
        #    wav = self.get_audio(i)
        #    self.audios.append(wav)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def load_audio(manifest_path, max_keep, min_keep):
        
        return root, names, inds, tot, sizes

    def batch_by_size(self, indices, max_tokens=None, max_sentences=None, required_batch_size_multiple=1):
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.required_batch_size_multiple = required_batch_size_multiple
        print(indices)
        if isinstance(indices[0], list):
            batch_list = []
            for indice in indices:
                batch = super(HubertDataset, self).batch_by_size(indice, max_tokens, max_sentences, required_batch_size_multiple)
                batch_list.append(batch)
            return batch_list
        else:
            return super(HubertDataset, self).batch_by_size(indices, max_tokens, max_sentences, required_batch_size_multiple)

    def shuffle_batches(self, batches, seed):
        if isinstance(batches[0], list):
            new_batches = []
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
                for batch in batches:
                    np.random.shuffle(batch)
                    new_batches.extend(batch)
            return new_batches
        else:
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
        return batches

    def reset_batch_sampler(self):
        indices = self.ordered_indices()
        batch_sampler = self.batch_by_size(
                indices,
                self.max_tokens,
                self.max_sentences,
                self.required_batch_size_multiple
        )
        return batch_sampler

    def get_noise_rir(self):
        with data_utils.numpy_seed(self.epoch):
            noise_id = np.random.choice(len(self.noise_names), 1)[0]
            rir_id = np.random.choice(len(self.rir_names), 1)[0]


            noise, sample_rate = sf.read(os.path.join(self.noise_root, self.noise_names[noise_id])) 
            rir, sample_rate = sf.read(os.path.join(self.rir_root, self.rir_names[noise_id]))
            snr = np.random.uniform(0,25)
            scale = np.random.uniform(0.3,0.9)
        return noise,rir,snr,scale

    def get_audio(self, index):
        import soundfile as sf

        wav_path = os.path.join(self.audio_root, self.audio_names[index])
        # wav_path = "/datablob/users/t-shren/data/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac"
        _path, slice_ptr = parse_path(wav_path)
        if len(slice_ptr) == 2:
            byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            # assert len(byte_data) > 3 , print(_path,slice_ptr[0],slice_ptr[1])
            assert is_sf_audio_data(byte_data)
            wav_path = io.BytesIO(byte_data)
        try:
            wav, cur_sample_rate = sf.read(wav_path)
        except:
            wav, cur_sample_rate = sf.read(wav_path)
        if self.add_noise:
            noise,rir,snr,scale = self.get_noise_rir()
            wav,_ = addscale(wav,scale)
            wav, _ = addreverb(wav,rir)            
            wav,_ = addnoise(wav,noise, snr, scale)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, cur_sample_rate)
        assert wav is not None, "wav is None, "+str(wav_path)
        return wav

    def get_norep_phoneme(self, phoneme_token):
        phoneme_token_no_rep = torch.from_numpy(
            np.array( 
                [ 
                    int(phoneme_token[i]) for i in range(1,len(phoneme_token)) if (phoneme_token[i] > 10) and (i==1 or phoneme_token[i]!=phoneme_token[i-1]) 
                ] 
            )
        )
        return phoneme_token_no_rep

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)

        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        labels = [self.get_label(index, i) for i in range(self.num_labels)]
        
        return [self.get_label(index, i) for i in range(self.num_labels)] 

    def __getitem__(self, index):
        # start = time.time()
        if len(self.audios) > index:
            wav = self.audios[index]
        else:
            wav = self.get_audio(index)
        # end = time.time()
       #  print("get audio time: ", str(end-start), str(self.PHF_INDEX))
        # start = time.time()
        labels = self.get_labels(index)
        # end = time.time()
        # print("get label time: ", str(end-start))
        if len(self.bnds) > 0:
            bnd = self.bnds[index]
        else:
            bnd = []
        
        return {"id": index, "source": wav, "label_list": labels, "boundary": bnd}

    def __len__(self):
        return len(self.sizes)

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

    def phoneme_padding_mask(self, phoneme_target):
        phoneme_sizes = [ len(s) for s in phoneme_target]
        # print(phoneme_sizes)
        max_size = max(phoneme_sizes)
        batch_size = len(phoneme_target)
        # print((batch_size,max_size))
        padd_mask = torch.zeros((batch_size, max_size)).bool()
        for  i, phoneme in enumerate(phoneme_target):
            diff =  max_size - len(phoneme) 
            if diff == 0:
                continue
            elif diff < 0:
                padd_mask[i, diff:] = True
        return padd_mask

    def get_accum_from_phoneme_seq(self, phoneme_seq, phoneme_padding_mask):
        bsz = phoneme_seq.shape[0]
        accum_lists = []
        for i in range(bsz):
            accum = [indice+1 for indice,j in enumerate(range(phoneme_seq[i].shape[0]-1)) 
                if phoneme_padding_mask[i][j] == False and phoneme_seq[i][j]!=phoneme_seq[i][j+1] ]
            accum_lists.append(accum)
        return accum_lists


    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples_bak = [s for s in samples if s["source"] is not None]
        if len(samples_bak) == 0:
            return {}
        samples = samples_bak
        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]
        bnds = [s["boundary"] for s in samples]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios, audio_size
        )

        targets_by_label = [
            [s["label_list"][i] for s in samples]
            for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )


        if self.PHF_INDEX != -1:
            if not self.is_valid:
                assert (max(audio_sizes) - 80) // 320 == targets_list[self.PHF_INDEX].shape[1] , str((max(audio_sizes) - 80) // 320) + " " +str(targets_list[self.PHF_INDEX].shape[1]) + " "+str([samples[i]["id"] for i in range(len(samples))])

            if self.is_valid: 
                net_input= {
                    "source": collated_audios, 
                    "padding_mask": padding_mask, 
                    "phf_input": None, 
                    "phf_padding_mask": None, 
                    "mode": "paired_data"
                }
            else:
            # if self.is_valid:
            #     print("****************************************")
            #     print(targets_list[self.PHF_INDEX].shape)
            #     print(self.phoneme_padding_mask(targets_by_label[self.PHF_INDEX]).shape )
            #     print("****************************************")
                accum_list = self.get_accum_from_phoneme_seq(targets_list[self.PHF_INDEX], self.phoneme_padding_mask(targets_by_label[self.PHF_INDEX]))

                net_input = {
                    "source": collated_audios, 
                    "padding_mask": padding_mask, 
                    "phf_input":targets_list[self.PHF_INDEX], 
                    "phf_padding_mask":  self.phoneme_padding_mask(targets_by_label[self.PHF_INDEX]),
                    "accum_list": accum_list,
                    "mode": "paired_data"
                }
        else:
            net_input = {
                "source": collated_audios, 
                "padding_mask": padding_mask, 
                "boundary": bnds,
                "mode": "speech_only"
            }
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }
        

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = [lengths_list[0]]
            batch["ntokens_list"] = [ntokens_list[0]]
            batch["target_list"] = [targets_list[0]]
        

        if self.multitask:
            batch["task"] = "multitask"
        else:
            batch["task"] = "hubert"
        return batch

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

    def collater_frm_label(
        self, targets, audio_size, audio_starts, label_rate, pad
    ):
        assert label_rate > 0
        s2f = label_rate / self.sample_rate
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s: s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            # print(label_rate)
            if label_rate == -1:
                targets, lengths, ntokens = self.collater_seq_label(
                    targets, pad
                )
            else:
                targets, lengths, ntokens = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        
        if self.shuffle:
            if len(self.chunk_names) > 0:
                with data_utils.numpy_seed(self.epoch):
                    self.chunk_order = np.random.permutation(len(self.chunk_names))
                chunk_count = 0
                tmp_sizes = []
                tmp_indices = []
                indice = []
                for i in self.chunk_order:
                    chunk_count += 1
                    start = self.chunk_indices[i]
                    end = self.chunk_indices[i+1] if i < len(self.chunk_names) - 1 else len(self)
                    size = list(self.sizes[start:end])
                    tmp_indices.extend(list(np.arange(start, end)))
                    tmp_sizes.extend(size)
                    if chunk_count % 1 == 0 or i == self.chunk_order[0]:
                        order = [np.random.permutation(len(tmp_indices))]
                        order.append(
                            np.minimum(
                                np.array(tmp_sizes),
                                self.max_sample_size,
                            )
                        )
                        sort_idx = np.lexsort(order)[::-1]
                        indice.append([tmp_indices[k] for k in sort_idx])
                        tmp_indices = []
                        tmp_sizes =[]
                return indice
            else:
                order = [np.random.permutation(len(self))]
                order.append(
                    np.minimum(
                        np.array(self.sizes),
                        self.max_sample_size,
                    )
                )
                return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))


    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav
