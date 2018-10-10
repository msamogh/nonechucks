from functools import partial
from future.utils import with_metaclass

import torch.utils.data as data

from .sampler import SafeSampler


class _SafeDataLoaderCaller(type):
    """Metaclass that overrides the __call__ method to replace
    `SequentialSampler` and `RandomSampler` with their corresponding
    `SafeSampler`s in DataLoader's namespace.
    """

    def __call__(cls, *args, **kwargs):
        cls.replace_default_samplers()
        obj = type.__call__(cls, *args, **kwargs)
        cls.restore_default_samplers()
        return obj

    def replace_default_samplers(cls):
        cls.sequential = data.dataloader.SequentialSampler
        cls.random = data.dataloader.RandomSampler

        def safe_sampler_callable(sampler_cls, dataset):
            return SafeSampler(dataset, sampler_cls(dataset))

        data.dataloader.SequentialSampler = partial(
            safe_sampler_callable, data.SequentialSampler)
        data.dataloader.RandomSampler = partial(
            safe_sampler_callable, data.RandomSampler)

    def restore_default_samplers(cls):
        data.dataloader.SequentialSampler = cls.sequential
        data.dataloader.RandomSampler = cls.random


class SafeDataLoader(with_metaclass(_SafeDataLoaderCaller, data.DataLoader)):
    """A DataLoader that reverts to safe versions of `SequentialSampler` and
    `RandomSampler` when no default sampler is specified.
    """
    pass
