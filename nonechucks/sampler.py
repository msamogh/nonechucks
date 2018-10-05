import torch
import torch.utils.data

from .dataset import SafeDataset


class SafeSampler(torch.utils.data.Sampler):
    """SafeSampler can be used both as a standard Sampler (over a Dataset),
    or as a wrapper around an existing `Sampler` instance. It allows you to
    drop unwanted samples while sampling.
    """

    def __init__(self, dataset, sampler=None, step_to_index_fn=None):
        """Create a `SafeSampler` instance that performs sampling over either
        another sampler object or directly over a dataset. `step_to_index_fn`
        will define the `SafeSampler` instance's behavior when it encounters
        an unsafe sample.

        When a `Sampler` object is passed to `sampler', the returned
        `SafeSampler` instance performs sampling over the list of indices
        sampled by `sampler`, acting as a wrapper over it. If `sampler` is
        `None`, the `SafeSampler` instance that is returned performs sampling
        directly over the `dataset` object.

        Arguments:
            dataset (SafeDataset): The dataset to be sampled.
            sampler (Sampler, optional): If `sampler` is `None`, the sampling
                is performed directly over the `dataset`, otherwise it's done
                over the list of indices returned by `sampler`'s `__iter__`
                method.
                If `sampler` takes a `Dataset` object as a parameter,
                `dataset` should ideally be the same as the one passed to
                `sampler`.
            step_to_index_fn (function, optional): Function that takes in 2
                arguments - (`num_valid_samples` and `num_samples_examined`),
                and returns the next index to be sampled. If None or not
                specified, the default function returns the
                `num_samples_examined` as the output.
        """
        assert isinstance(dataset, SafeDataset), \
            "Dataset must be a SafeDataset."
        self.dataset = dataset
        self.dataset.__class__.__getitem__ = self.dataset._get_sample_original

        self.sampler = sampler
        if sampler is not None:
            self.sampler_indices = list(iter(sampler))

        if step_to_index_fn is not None:
            self.step_to_index_fn = step_to_index_fn
        else:
            self.step_to_index_fn = lambda original, actual: actual

    def __iter__(self):
        """Return iterator over sampled indices."""
        self.num_valid_samples = self.num_samples_examined = 0
        return self

    def _get_next_index(self):
        """Helper function that calls `step_to_index_fn` and decides
        whether to sample directly from `dataset` or through `sampler`."""
        index = self.step_to_index_fn(
            self.num_valid_samples, self.num_samples_examined)
        if self.sampler is not None:
            index = self.sampler_indices[index]
        return index

    def __next__(self):
        """Returns next index to sample over `dataset`."""
        while True:
            try:
                index = self._get_next_index()
                self.num_samples_examined += 1
                if self.dataset._safe_get_item(index) is not None:
                    self.num_valid_samples += 1
                    return index
            except IndexError:
                raise StopIteration

    # For Python2 compatibility
    next = __next__
