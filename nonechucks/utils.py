import collections

from itertools import chain
from functools import partial

import torch
try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data._utils.collate import default_collate
from torch._six import string_classes


class memoize(object):
    """cache the return value of a method

    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.

    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    class Obj(object):
        @memoize
        def add_to(self, arg):
            return self + arg
    Obj.add_to(1) # not enough arguments
    Obj.add_to(1, 2) # returns 3, result is not cached
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


def collate_batches(batches, collate_fn=default_collate):
    """Collate multiple batches."""
    error_msg = "batches must be tensors, dicts, or lists; found {}"
    if isinstance(batches[0], torch.Tensor):
        return torch.cat(batches, 0)
    elif isinstance(batches[0], collections.Sequence):
        return list(chain(*batches))
    elif isinstance(batches[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batches])
                for key in batches[0]}
    raise TypeError((error_msg.format(type(batches[0]))))


def batch_len(batch):
    # error_msg = "batch must be tensor, dict, or list: found {}"
    if isinstance(batch, list):
        if isinstance(batch[0], string_classes):
            return len(batch)
        else:
            return len(batch[0])
    elif isinstance(batch, collections.Mapping):
        first_key = list(batch.keys())[0]
        return len(batch[first_key])
    return len(batch)


def slice_batch(batch, start=None, end=None):
    if isinstance(batch, list):
        if isinstance(batch[0], string_classes):
            return batch[start:end]
        else:
            return [sample[start:end] for sample in batch]
    elif isinstance(batch[0], collections.Mapping):
        return {key: batch[key][start:end] for key in batch}
    else:
        return batch[start:end]
