from functools import partial
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.testing._internal.common_utils import TestCase

import nonechucks as nc

DATASET_SIZE = 17

class TensorDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, idx):
        return self.tensors[idx]

    def __len__(self):
        return len(self.tensors)

class DictDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, idx):
        tensor = self.tensors[idx]
        return dict(idx=idx, tensor=tensor) if tensor else None

    def __len__(self):
        return len(self.tensors)


class TestSafeDataLoader(TestCase):
    def test_alright(self):
        r"""Checks that the dataset is correctly loaded when nothing is wrong"""

        tensors = [torch.randn(2,3) for _ in range(DATASET_SIZE)]
        dataset = nc.SafeDataset(TensorDataset(tensors))

        for batch_size in range(1, DATASET_SIZE):
            loader = nc.SafeDataLoader(dataset, batch_size=batch_size)

            not_seen = DATASET_SIZE
            for batch in loader:
                self.assertEqual(batch.size(0), min(batch_size, not_seen))
                not_seen -= batch.size(0)


    def test_none(self):
        r"""Checks that `None`s in the dataset are ignored"""

        for num_nones in range(1, DATASET_SIZE):
            tensors = [torch.randn(2,3) for _ in range(DATASET_SIZE)]
            for i in random.sample(list(range(DATASET_SIZE)), num_nones):
                tensors[i] = None

            dataset = nc.SafeDataset(TensorDataset(tensors))

            for batch_size in range(1, DATASET_SIZE):
                loader = nc.SafeDataLoader(dataset, batch_size=batch_size)

                not_seen = DATASET_SIZE - num_nones
                for batch in loader:
                    self.assertEqual(batch.size(0), min(batch_size, not_seen))
                    not_seen -= batch.size(0)


class TestCollate(TestCase):
    def test_custom_collate_alright(self):
        r"""Custom collate_fn when whole dataset is valid"""

        tensors = [torch.randn(2,3) for _ in range(DATASET_SIZE)]
        dataset = nc.SafeDataset(TensorDataset(tensors))

        for batch_size in range(1, DATASET_SIZE):
            loader = nc.SafeDataLoader(dataset, batch_size=batch_size, collate_fn=torch.stack)

            not_seen = DATASET_SIZE
            for batch in loader:
                self.assertEqual(batch.size(0), min(batch_size, not_seen))
                not_seen -= batch.size(0)

    def test_custom_collate_none(self):
        r"""Custom collate_fn when one sample is corrupted"""

        tensors = [torch.randn(2,3) for _ in range(DATASET_SIZE)]
        tensors[5] = None
        dataset = nc.SafeDataset(TensorDataset(tensors))

        for batch_size in range(1, DATASET_SIZE):
            loader = nc.SafeDataLoader(dataset, batch_size=batch_size, collate_fn=torch.stack)

            not_seen = DATASET_SIZE - 1
            for batch in loader:
                self.assertEqual(batch.size(0), min(batch_size, not_seen))
                not_seen -= batch.size(0)

    def test_padding_alright(self):
        r"""Pads sequence of different-sizes tensors"""
        tensors = [torch.randn(2*i+1, 3) for i in range(DATASET_SIZE)]
        dataset = nc.SafeDataset(TensorDataset(tensors))

        for batch_size in range(1, DATASET_SIZE):
            loader = nc.SafeDataLoader(dataset, batch_size=batch_size, collate_fn=pad_sequence)

            not_seen = DATASET_SIZE
            for batch in loader:
                self.assertEqual(batch.size(1), min(batch_size, not_seen))
                not_seen -= batch.size(1)

    def test_padding_none(self):
        r"""Pads sequence of different-sizes tensors"""
        tensors = [torch.randn(2*i+1, 3) for i in range(DATASET_SIZE)]
        tensors[5] = None
        dataset = nc.SafeDataset(TensorDataset(tensors))

        for batch_size in range(1, DATASET_SIZE):
            loader = nc.SafeDataLoader(dataset, batch_size=batch_size, collate_fn=pad_sequence)
            not_seen = DATASET_SIZE - 1
            for batch in loader:

                self.assertEqual(batch.size(1), min(batch_size, not_seen))
                not_seen -= batch.size(1)

    def test_padding_alright_batch_first(self):
        r"""Same as above with batch first"""
        tensors = [torch.randn(2*i+1, 3) for i in range(DATASET_SIZE)]
        dataset = nc.SafeDataset(TensorDataset(tensors))

        for batch_size in range(1, DATASET_SIZE):
            loader = nc.SafeDataLoader(dataset, batch_size=batch_size, collate_fn=partial(pad_sequence, batch_first=True))

            not_seen = DATASET_SIZE
            for batch in loader:
                self.assertEqual(batch.size(0), min(batch_size, not_seen))
                not_seen -= batch.size(0)

    def test_padding_none_batch_first(self):
        r"""Same as above with batch first"""
        tensors = [torch.randn(2*i+1, 3) for i in range(DATASET_SIZE)]
        tensors[5] = None
        dataset = nc.SafeDataset(TensorDataset(tensors))

        for batch_size in range(1, DATASET_SIZE):
            loader = nc.SafeDataLoader(dataset, batch_size=batch_size, collate_fn=partial(pad_sequence, batch_first=True))
            not_seen = DATASET_SIZE - 1
            for batch in loader:
                self.assertEqual(batch.size(0), min(batch_size, not_seen))
                not_seen -= batch.size(0)




class TestDictDataset(TestCase):
    def test_alright(self):
        r"""Elements of the dataset are dicts"""
        tensors = [torch.randn(2,3) for _ in range(DATASET_SIZE)]
        dataset = nc.SafeDataset(DictDataset(tensors))

        for batch_size in range(1, DATASET_SIZE):
            loader = nc.SafeDataLoader(dataset, batch_size=batch_size)

            not_seen = DATASET_SIZE
            for batch in loader:
                self.assertEqual(batch['idx'].size(0), min(batch_size, not_seen))
                self.assertEqual(batch['tensor'].size(0), min(batch_size, not_seen))
                not_seen -= batch['idx'].size(0)

    def test_none(self):
        tensors = [torch.randn(2,3) for _ in range(DATASET_SIZE)]
        tensors[5] = None
        dataset = nc.SafeDataset(DictDataset(tensors))

        for batch_size in range(1, DATASET_SIZE):
            loader = nc.SafeDataLoader(dataset, batch_size=batch_size)

            not_seen = DATASET_SIZE - 1
            for batch in loader:
                self.assertEqual(batch['idx'].size(0), min(batch_size, not_seen))
                self.assertEqual(batch['tensor'].size(0), min(batch_size, not_seen))
                not_seen -= batch['idx'].size(0)

    def test_padding_alright(self):
        tensors = [torch.randn(2*i+1, 3) for i in range(DATASET_SIZE)]
        dataset = nc.SafeDataset(DictDataset(tensors))

        for batch_size in range(1, DATASET_SIZE):
            loader = nc.SafeDataLoader(dataset, batch_size=batch_size, collate_fn=self.pad)

            not_seen = DATASET_SIZE
            for batch in loader:
                self.assertEqual(batch.size(1), min(batch_size, not_seen))
                not_seen -= batch.size(1)

    def test_padding_none(self):
        tensors = [torch.randn(2*i+1, 3) for i in range(DATASET_SIZE)]
        tensors[5] = None
        dataset = nc.SafeDataset(DictDataset(tensors))

        for batch_size in range(1, DATASET_SIZE):
            loader = nc.SafeDataLoader(dataset, batch_size=batch_size, collate_fn=self.pad)
            not_seen = DATASET_SIZE - 1
            for batch in loader:
                self.assertEqual(batch.size(1), min(batch_size, not_seen))
                not_seen -= batch.size(1)

    @staticmethod
    def pad(batch):
        if len(batch) == 0:
            return {}
        return {key: pad_sequence(batch[key]) for key in batch[0].keys()}
