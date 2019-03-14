from functools import partial
from future.utils import with_metaclass

import torch.utils.data as data
try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data._utils.collate import default_collate

from .dataset import SafeDataset
from .sampler import SafeSampler
from .utils import batch_len, collate_batches, slice_batch


class _SafeDataLoaderCaller(type):
    """Metaclass that overrides the __call__ method to replace
    `SequentialSampler` and `RandomSampler` with their corresponding
    `SafeSampler`s in DataLoader's namespace.
    """

    def __call__(cls, *args, **kwargs):
        cls._replace_default_samplers()
        obj = type.__call__(cls, *args, **kwargs)
        cls._restore_default_samplers()
        return obj

    def _replace_default_samplers(cls):
        cls.sequential = data.dataloader.SequentialSampler
        cls.random = data.dataloader.RandomSampler

        def safe_sampler_callable(sampler_cls, dataset):
            return SafeSampler(dataset, sampler_cls(dataset))

        data.dataloader.SequentialSampler = partial(
            safe_sampler_callable, data.SequentialSampler)
        data.dataloader.RandomSampler = partial(
            safe_sampler_callable, data.RandomSampler)

    def _restore_default_samplers(cls):
        data.dataloader.SequentialSampler = cls.sequential
        data.dataloader.RandomSampler = cls.random


class _SafeDataLoaderIter(data.dataloader._DataLoaderIter):

    def __init__(self, loader):
        super().__init__(loader)
        self.batch_size = loader.batch_size
        self.coalescing_in_progress = False
        self.drop_last = loader.drop_last_original
        if isinstance(loader.sampler, SafeSampler):
            self.step_to_index_fn = loader.sampler.step_to_index_fn
        else:
            self.step_to_index_fn = SafeSampler.default_step_to_index_fn

    def _process_next_batch(self, curr_batch):
        """Fills an incomplete batch if necessary before processing it."""
        super()._process_next_batch(curr_batch)
        if self.coalescing_in_progress:
            return curr_batch

        # When set to True, _process_next_batch simply returns an incomplete
        # batch instead of trying to fill it up to a length of batch_size.
        #
        # Stays True until the current batch is filled so that nested calls for
        # _process_next_batch from within the loop simply call the parent
        # method.
        self.coalescing_in_progress = True
        n_empty_slots = self.batch_size - batch_len(curr_batch)
        while n_empty_slots > 0:
            # check if curr_batch is the final batch
            if self.batches_outstanding == 0 and not self.reorder_dict:
                if (not self.drop_last) or \
                   (batch_len(curr_batch) == self.batch_size):
                    return curr_batch

            # raises StopIteration if no more elements left, which exits the
            # loop
            next_batch = next(self)
            if len(next_batch) == 0:
                super()._process_next_batch(next_batch)
                continue
            elif len(next_batch) > n_empty_slots:
                # Take only n_empty_slots number of samples from next_batch.
                # The remaining elements of next_batch are added back into the
                # dict for future consumption.
                self.rcvd_idx -= 1
                curr_batch = collate_batches([
                    curr_batch,
                    slice_batch(next_batch, end=n_empty_slots)
                ])
                self.reorder_dict[self.rcvd_idx] = slice_batch(
                    next_batch,
                    start=n_empty_slots
                )
            else:
                curr_batch = collate_batches([curr_batch, next_batch])

            n_empty_slots -= min(
                n_empty_slots,
                batch_len(next_batch)
            )
        self.coalescing_in_progress = False
        return curr_batch


class _OriginalDataset(SafeDataset):
    """Wraps a SafeDataset to return None for invalid samples."""

    def __init__(self, safe_dataset):
        self.safe_dataset = safe_dataset

    def __getitem__(self, idx):
        return self.safe_dataset._safe_get_item(idx)


class SafeDataLoader(with_metaclass(_SafeDataLoaderCaller, data.DataLoader)):
    """A DataLoader that reverts to safe versions of `SequentialSampler` and
    `RandomSampler` when no default sampler is specified.
    """

    @staticmethod
    def _safe_default_collate(batch):
        filtered_batch = [x for x in batch if x is not None]
        if len(filtered_batch) == 0:
            return []
        return default_collate(filtered_batch)

    def __init__(self, dataset, **kwargs):
        # drop_last is handled transparently by _SafeDataLoaderIter (bypassing
        # DataLoader). Since drop_last cannot be changed after initializing the
        # DataLoader instance, it needs to be intercepted here.
        assert isinstance(dataset, SafeDataset), \
            "dataset must be an instance of SafeDataset."

        self.drop_last_original = False
        if 'drop_last' in kwargs:
            self.drop_last_original = kwargs['drop_last']
            kwargs['drop_last'] = False
        super(SafeDataLoader, self).__init__(dataset, **kwargs)

        self.safe_dataset = self.dataset
        self.dataset = _OriginalDataset(self.safe_dataset)

        if self.collate_fn is default_collate:
            self.collate_fn = SafeDataLoader._safe_default_collate

    def __iter__(self):
        if self.num_workers > 0:
            return _SafeDataLoaderIter(self)
        return data.dataloader._DataLoaderIter(self)
