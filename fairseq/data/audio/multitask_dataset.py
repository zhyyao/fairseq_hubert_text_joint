# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import bisect

import logging
import numpy as np
from torch.utils.data.dataloader import default_collate


from fairseq.data import FairseqDataset
from fairseq.data import data_utils

logger = logging.getLogger(__name__)

class MultitaskDataset(FairseqDataset):
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            #print(len(e))
            #print(e)
            curr_len = len(e)
            r.append(curr_len + s)
            s += curr_len
        return r

    def __init__(self, datasets, sample_ratios=1, batch_ratio=None, seed=1337,world_size=1):
        super(MultitaskDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        if isinstance(sample_ratios, int):
            sample_ratios = [sample_ratios] * len(self.datasets)
            self.batch_ratio=None
        else:
            logger.info('set sample ratio to ' + str(sample_ratios))
            if batch_ratio is not None:
                logger.info('batch ratio is ' + str(batch_ratio))
                self.batch_ratio = batch_ratio
            else:
                self.batch_ratio = None
        self.sample_ratios = sample_ratios
        self._ordered_indices = None
        self.max_tokens = []
        self.max_sentences = []
        self.required_batch_size_multiple = []
        self._update_size()
        self.seed = seed
        self.world_size = world_size

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        sample = self.datasets[dataset_idx][sample_idx]
        if isinstance(sample, dict):
            sample["dataset_idx"] = dataset_idx
        else:
            sample = sample + (dataset_idx,)
        return sample
    @property
    def supports_fetch_outside_dataloader(self):
        """Whether this dataset supports fetching outside the workers of the dataloader."""
        return False

    def _update_size(self):
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.real_sizes = [len(d) for d in self.datasets]

    def _get_dataset_and_sample_index(self, idx: int):
        #print(self.cumulative_sizes)
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]
        return dataset_idx, sample_idx

    def collater(self, samples, **extra_args):
        # For now only supports datasets with same underlying collater implementations
        if samples is not None and len(samples) > 0:
            if isinstance(samples[0], dict):
                dataset_idx = samples[0]["dataset_idx"]
            else:
                dataset_idx = samples[0][-1]
                samples = [sample[:-1] for sample in samples]
        else:
            dataset_idx = 0

        if hasattr(self.datasets[dataset_idx], "collater"):
            return self.datasets[dataset_idx].collater(samples, **extra_args)
        else:
            return default_collate(samples, **extra_args)

    def size(self, idx: int):
        """
        Return an example's size as a float or tuple.
        """
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[dataset_idx].size(sample_idx)

    def num_tokens(self, index: int):
        return np.max(self.size(index))

    def attr(self, attr: str, index: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        return getattr(self.datasets[dataset_idx], attr, None)

    @property
    def sizes(self):
        _dataset_sizes = []
        for ds in self.datasets:
            if isinstance(ds.sizes, np.ndarray):
                _dataset_sizes.append(ds.sizes)
            else:
                # Only support underlying dataset with single size array.
                assert isinstance(ds.sizes, list)
                _dataset_sizes.append(ds.sizes[0])
        return np.concatenate(_dataset_sizes)

    @property
    def supports_prefetch(self):
        return all(d.supports_prefetch for d in self.datasets)

        
    def ordered_indices(self):
        # if self._ordered_indices is None:
            # Call the underlying dataset's ordered_indices() here, so that we
            # get the same random ordering as we would have from using the
            # underlying sub-datasets directly.
        self._ordered_indices = [
            dataset.ordered_indices()
            for dataset in self.datasets
        ]
        
        return np.arange(len(self))

    def prefetch(self, indices):
        frm = 0
        for to, ds in zip(self.cumulative_sizes, self.datasets):
            real_size = len(ds)
            if getattr(ds, "supports_prefetch", False):
                ds.prefetch([(i - frm) % real_size for i in indices if frm <= i < to])
            frm = to

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        batch_samplers = []
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.required_batch_size_multiple = required_batch_size_multiple
        for i, dataset in enumerate(self.datasets):
            batch_sampler = dataset.batch_by_size(
                self._ordered_indices[i],
                max_tokens=max_tokens if self.batch_ratio is None else max_tokens * self.batch_ratio[i],
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
            if i > 0:
                for batch in batch_sampler:
                    batch += self.cumulative_sizes[i - 1]
            if self.sample_ratios[i] != 1.0:
                with data_utils.numpy_seed(self.seed + self.epoch):
                    batch_sampler = np.array(batch_sampler)
                    batch_sampler = np.random.choice(batch_sampler, int(len(batch_sampler) * self.sample_ratios[i]))
                    batch_sampler = list(batch_sampler)
            batch_samplers.extend(batch_sampler)

        
        return batch_samplers

    # !!!!!!!!!!!!!!!!!
    # This function only used for speech text multitask dataset
    # !!!!!!!!!!!!!!!!!
    def shuffle_batches(self,batches,seed):
        if False:
            new_batches = []
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
                for batch in batches:
                    if isinstance(batch[0], np.ndarray):
                        np.random.shuffle(batch)
                        new_batches.extend(batch)
                    else:
                        new_batches.append(batch)
        else:
            data_batches = [[] for i in range(len(self.datasets))]
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
                for batch in batches:
                    if isinstance(batch[0], np.ndarray):
                        np.random.shuffle(batch)
                        data_batches[0].extend(batch)
                    else:
                        data_batches[ self._get_dataset_and_sample_index(batch[0])[0] ].append(batch)
            new_data_batches = []
            dataset_num = len(data_batches)
            if len(data_batches) == 1:
                new_data_batches = data_batches[0]
                return new_data_batches
            size = self.world_size
            end = [0 for _ in range(dataset_num)]
            for i in range(dataset_num):
                for j in range(len(data_batches[i]) // size):
                    new_data_batches.append( data_batches[i][j *size : j*size+size] )
                    end[i]=j*size +size
            new_batches = []
            with data_utils.numpy_seed(seed):
                np.random.shuffle(new_data_batches)
                for batch in new_data_batches:
                    new_batches.extend(batch)
            for i in range(dataset_num):
                new_batches.extend(data_batches[i][end[i]:])            
        return new_batches

    # def reset_batch_sampler(self):
    #     indices = self.ordered_indices
    #     batch_sampler = self.batch_by_size(
    #             indices,
    #             self.max_tokens,
    #             self.max_sentences,
    #             self.required_batch_size_multiple
    #     )
    #     return batch_sampler
              
    def filter_indices_by_size(self, indices, max_positions):
        """
        Filter each sub-dataset independently, then update the round robin to work
        on the filtered sub-datasets.
        """

        ignored_some = False
        for i in range(len(self.datasets)):
            # dataset = _deep_until_language_pair(dataset)
            self._ordered_indices[i], ignored = self.datasets[i].filter_indices_by_size(
                self._ordered_indices[i], max_positions * self.batch_ratio[i]
            )
            if len(ignored) > 0:
                ignored_some = True
                logger.warning(
                    f"{len(ignored)} samples from {i} have invalid sizes and will be skipped, "
                    f"max_positions={max_positions}, first few sample ids={ignored[:10]}"
                )

        logger.info('update dataset size')
        self._update_size()

        # Since we are modifying in place the _ordered_indices,
        # it's not possible anymore to return valid ignored indices.
        # Hopefully the extra debug information print above should be enough to debug.
        # Ideally we would receive ignore_invalid_inputs so that we could have
        # a proper error message.
        return (np.arange(len(self)), [0] if ignored_some else [])

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

    

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.epoch = epoch
        for ds in self.datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)
