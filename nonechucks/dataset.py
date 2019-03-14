import torch
import torch.utils.data

from .utils import memoize


class SafeDataset(torch.utils.data.Dataset):
    """A wrapper around a torch.utils.data.Dataset that allows dropping
    samples dynamically.
    """

    def __init__(self, dataset, eager_eval=False):
        """Creates a `SafeDataset` wrapper around `dataset`."""
        self.dataset = dataset
        self.eager_eval = eager_eval
        # These will contain indices over the original dataset. The indices of
        # the safe samples will go into _safe_indices and similarly for unsafe
        # samples.
        self._safe_indices = []
        self._unsafe_indices = []

        # If eager_eval is True, we can simply go ahead and build the index
        # by attempting to access every sample in self.dataset.
        if self.eager_eval is True:
            self._build_index()

    def _safe_get_item(self, idx):
        """Returns None instead of throwing an error when dealing with an
        unsafe sample, and also builds an index of safe and unsafe samples as
        and when they get accessed.
        """
        try:
            # differentiates IndexError occuring here from one occuring during
            # sample loading
            invalid_idx = False
            if idx >= len(self.dataset):
                invalid_idx = True
                raise IndexError
            sample = self.dataset[idx]
            if idx not in self._safe_indices:
                self._safe_indices.append(idx)
            return sample
        except Exception as e:
            if isinstance(e, IndexError):
                if invalid_idx:
                    raise
            if idx not in self._unsafe_indices:
                self._unsafe_indices.append(idx)
            return None

    def _build_index(self):
        for idx in range(len(self.dataset)):
            # The returned sample is deliberately discarded because
            # self._safe_get_item(idx) is called only to classify every index
            # into either safe_samples_indices or _unsafe_samples_indices.
            _ = self._safe_get_item(idx)

    def _reset_index(self):
        """Resets the safe and unsafe samples indices."""
        self._safe_indices = self._unsafe_indices = []

    @property
    def is_index_built(self):
        """Returns True if all indices of the original dataset have been
        classified into safe_samples_indices or _unsafe_samples_indices.
        """
        return len(self.dataset) == len(self._safe_indices) + \
            len(self._unsafe_indices)

    @property
    def num_samples_examined(self):
        return len(self._safe_indices) + len(self._unsafe_indices)

    def __len__(self):
        """Returns the length of the original dataset.
        NOTE: This is different from the number of actually valid samples.
        """
        return len(self.dataset)

    def __iter__(self):
        return (self._safe_get_item(i) for i in range(len(self))
                if self._safe_get_item(i) is not None)

    @memoize
    def __getitem__(self, idx):
        """Behaves like the standard __getitem__ for Dataset when the index
        has been built.
        """
        while idx < len(self.dataset):
            sample = self._safe_get_item(idx)
            if sample is not None:
                return sample
            idx += 1
        raise IndexError
