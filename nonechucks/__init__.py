import logging

import torch
import torch.utils.data


logger = logging.getLogger(__name__)


def _get_pytorch_version():
    version = torch.__version__
    if '+' in version:
        # e.g. 1.6.0+cu101
        version = version[:version.index('+')]
    major, minor = [int(x) for x in version.split(".")[:2]]
    if major != 1:
        raise RuntimeError(
            "nonechucks only supports PyTorch major version 1 at the moment."
        )
    if minor > 7:
        logger.warn(
            "nonechucks may not work properly with this version of PyTorch ({}). "
            "It has only been tested on PyTorch versions up to 1.7".format(
                version
            )
        )
    return major, minor


MAJOR, MINOR = _get_pytorch_version()

if MINOR > 1:
    SingleProcessDataLoaderIter = (
        torch.utils.data.dataloader._SingleProcessDataLoaderIter
    )
    MultiProcessingDataLoaderIter = (
        torch.utils.data.dataloader._MultiProcessingDataLoaderIter
    )
else:
    SingleProcessDataLoaderIter = torch.utils.data.dataloader._DataLoaderIter
    MultiProcessingDataLoaderIter = torch.utils.data.dataloader._DataLoaderIter


from nonechucks.dataset import SafeDataset
from nonechucks.sampler import SafeSampler
from nonechucks.dataloader import SafeDataLoader
