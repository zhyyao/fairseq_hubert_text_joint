from functools import lru_cache

import numpy as np
import torch
from fairseq.data import Dictionary, data_utils

from . import BaseWrapperDataset, LRUCacheDataset

class TextRepDataset(BaseWrapperDataset):
    """
    A wrapper Dataset 
    """
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        accum_path,
        seed: int = 1,       
    ):
        self.dataset=dataset
        self.vocab = vocab
        self.accum_stat, self.rep_list, self.rep_sz_list = self.load_accum_stat(accume_path)
        self.seed = seed
        self.epoch = 0

    def load_accum_stat(self, accum_path):
        accum_stat = {}
        str_map = {}
        rep_list = []
        sz_list = []
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
            if phoneme not in str_map.keys():
                str_map[phoneme] = len(rep_list)
                rep_list.append(phoneme * ( accum_stat[phoneme+"_B"] + \
                                            accum_stat[phoneme+"_I"] + \
                                            accum_stat[phoneme+"_E"] + \
                                            accum_stat[phoneme+"_S"] ))
                sz_list.append(accum_stat[phoneme+"_B"] + accum_stat[phoneme+"_I"] + accum_stat[phoneme+"_E"]+ accum_stat[phoneme+"_S"])
                
        return str_map, rep_list, sz_list

    def avoid_zero(self, accum_stat,key):
        prefix = key.split("_")[0]
        if accum_stat[prefix+"_B"] + accum_stat[prefix+"_I"] + accum_stat[prefix+"_E"] + accum_stat[prefix+"_S"] ==0:
            accum_stat[prefix+"_B"] =1 
            accum_stat[prefix+"_I"] =1
            accum_stat[prefix+"_E"] =0
            accum_stat[prefix+"_S"] =0

    def __getitem__(self, index):
        return self.__getitem_cached__(self.seed, self.epoch, index)

    