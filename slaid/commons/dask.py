import logging
import os

import dask.array as da
import tiledb
import zarr
from dask.distributed import Client, progress

from slaid.commons import Mask as BaseMask
from slaid.commons.base import Slide as BaseSlide, SlideArray

logger = logging.getLogger()


def init_client(address=None, processes=False):
    if address:
        return Client(address)
    else:
        return Client(processes=processes)


class Mask(BaseMask):
    def compute(self):
        if isinstance(self.array, da.Array):
            self.array = self.array.compute(rerun_exceptions_locally=True)

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
        if isinstance(self.array, da.Array):
            task = da.to_zarr(self.array,
                              path,
                              compute=True,
                              return_stored=True)
            progress(task)

        else:
            zarr.save(path, self.array)
        array = group[os.path.basename(path)]
        for attr, value in self._get_attributes().items():
            logger.info('writing attr %s %s', attr, value)
            array.attrs[attr] = value


class Slide(BaseSlide):
    def _read_from_store(self, dataset):
        return da.from_zarr(self._store, component=dataset["path"])

    def _create_slide(self, dataset):
        return DaskSlideArray(self._read_from_store(dataset),
                              self._slide.IMAGE_INFO).convert(self.image_info)


class DaskSlideArray(SlideArray):
    @property
    def array(self):
        return self._array
