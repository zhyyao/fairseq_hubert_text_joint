# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import BaseWrapperDataset


class ModeDataset(BaseWrapperDataset):
    def __init__(self, dataset, mode="text_only"):
        super().__init__(dataset)
        self.mode = mode

    def __getitem__(self, index):
        item = self.dataset[index]
        return self.mode

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return samples
