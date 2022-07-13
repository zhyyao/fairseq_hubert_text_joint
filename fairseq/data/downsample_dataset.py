# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import BaseWrapperDataset


class DownSampleDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return item[::2]

    def __len__(self):
        return len(self.dataset)

    @property
    def sizes(self):
        return self.dataset.sizes // 2

