import collections
import unittest

try:
    import unittest.mock as mock
except ImportError:
    import mock

import torch
import torch.utils.data as data

from nonechucks import *
import nonechucks


class SafeDatasetTest(unittest.TestCase):
    """Unit tests for `SafeDataset`."""

    SafeDatasetPair = collections.namedtuple(
        'SafeDatasetPair', ['unsafe', 'safe'])

    @classmethod
    def get_safe_dataset_pair(cls, dataset, **kwargs):
        """Returns a `SafeDatasetPair` (a tuple of size 2), which contains
            both the unsafe and safe versions of the dataset.
        """
        return SafeDatasetTest.SafeDatasetPair(
            dataset, nonechucks.SafeDataset(dataset, **kwargs))

    def setUp(self):
        tensor_data = data.TensorDataset(torch.arange(0, 10))
        self._dataset = self.get_safe_dataset_pair(tensor_data)

    @property
    def dataset(self):
        self._dataset.safe._reset_index()
        return self._dataset

    def test_build_index(self):
        dataset = data.TensorDataset(torch.arange(0, 10))
        dataset = self.get_safe_dataset_pair(dataset, eager_eval=True)

        self.assertTrue(dataset.safe.is_index_built)
        self.assertEqual(len(dataset.safe), len(dataset.unsafe))

    def test_dataset_iterator(self):
        counter = 0
        for i in self.dataset.safe:
            self.assertEqual(i[0].tolist(), counter)
            counter += 1

    def test_iter_calls_safe_get_item(self):
        dataset = data.TensorDataset(torch.arange(0, 10))
        dataset = self.get_safe_dataset_pair(dataset).safe
        for sample in dataset:
            pass
        self.assertTrue(dataset.is_index_built)

    # @mock.patch('torch.utils.data.TensorDataset.__getitem__')
    # def test_default_map(self, mock_get_item):
    #     def side_effect(idx):
    #         return [10, 11, 12, 13, None, 14, None, None, 15, 16][idx]
    #     mock_get_item.side_effect = side_effect

    #     dataset = data.TensorDataset(torch.arange(0, 10))
    #     dataset = self.get_safe_dataset_pair(dataset)
    #     self.assertEqual(dataset.safe[4], 14)
    #     self.assertEqual(dataset.safe[5], 15)
    #     self.assertEqual(dataset.safe[4], 14)

    def test_memoization(self):
        pass

    def test_import(self):
        self.assertIsNotNone(SafeDataset)
        self.assertIsNotNone(SafeSampler)

if __name__ == '__main__':
    unittest.main()
