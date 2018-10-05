import unittest

try:
    import unittest.mock as mock
except ImportError:
    import mock

import torch
import torch.utils.data as data

import nonechucks


class SafeSamplerTest(unittest.TestCase):

    def test_sequential_sampler(self):
        dataset = data.TensorDataset(torch.arange(0, 10))
        dataset = nonechucks.SafeDataset(dataset)
        dataloader = data.DataLoader(
            dataset,
            sampler=nonechucks.SafeSequentialSampler(dataset))
        for i_batch, sample_batched in enumerate(dataloader):
            print('Sample {}: {}'.format(i_batch, sample_batched))

    def test_first_last_sampler(self):
        dataset = data.TensorDataset(torch.arange(0, 10))
        dataset = nonechucks.SafeDataset(dataset)
        dataloader = data.DataLoader(
            dataset,
            sampler=nonechucks.SafeFirstAndLastSampler(dataset))
        for i_batch, sample_batched in enumerate(dataloader):
            print('Sample {}: {}'.format(i_batch, sample_batched))

    @mock.patch('torch.utils.data.TensorDataset.__getitem__')
    @mock.patch('torch.utils.data.TensorDataset.__len__')
    def test_sampler_wrapper(self, mock_len, mock_get_item):
        def side_effect(idx):
            return [0, 1, None, 3, 4, 5][idx]
        mock_get_item.side_effect = side_effect
        mock_len.return_value = 6
        dataset = data.TensorDataset(torch.arange(0, 10))
        dataset = nonechucks.SafeDataset(dataset)
        self.assertEqual(len(dataset), 6)
        sequential_sampler = data.SequentialSampler(dataset)
        dataloader = data.DataLoader(
            dataset,
            sampler=nonechucks.SafeSampler(dataset, sequential_sampler))
        for i_batch, sample_batched in enumerate(dataloader):
            print('Sample {}: {}'.format(i_batch, sample_batched))

if __name__ == '__main__':
    unittest.main()
