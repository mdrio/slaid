import logging
import os

import dask.array as da
import tiledb
import zarr
from dask.distributed import Client

from slaid.commons import Mask as BaseMask

logger = logging.getLogger()


def init_client(address=None, processes=False):
    if address:
        return Client(address)
    else:
        return Client(processes=processes)


class Mask(BaseMask):
    def to_tiledb(self, path: str, overwrite: bool = False, **kwargs):
        if overwrite:
            try:
                tiledb.remove(path)
            except tiledb.libtiledb.TileDBError as ex:
                logger.error(ex)
        da.to_tiledb(self.array, path, compute=True, **kwargs)
        self._write_meta_tiledb(path)

    #  FIXME: duplicate code
    def to_zarr(self, path: str, overwrite: bool = False, **kwargs):
        logger.info('dumping mask to zarr on path %s', path)
        name = os.path.basename(path)
        group = zarr.open_group(os.path.dirname(path))
        if overwrite and name in group:
            del group[name]
        da.to_zarr(self.array, path, compute=True)
        array = group[os.path.basename(path)]
        for attr, value in self._get_attributes().items():
            logger.info('writing attr %s %s', attr, value)
            array.attrs[attr] = value
