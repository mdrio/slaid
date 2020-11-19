import logging

import dask.array as da
from dask.distributed import Client

from slaid.commons import Mask as BaseMask

logger = logging.getLogger()


def init_client(*args, **kwargs):
    logger.debug('init dask client with %s, %s', args, kwargs)
    return Client(*args, **kwargs)


class Mask(BaseMask):
    def to_tiledb(self, path, **kwargs):
        da.to_tiledb(self.array, path)
        self._write_meta_tiledb(path)
