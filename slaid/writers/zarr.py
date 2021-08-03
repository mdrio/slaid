import logging
import os
from datetime import datetime as dt
from tempfile import TemporaryDirectory

import zarr

from slaid.commons import BasicSlide, Mask
from slaid.commons.ecvl import BasicSlide as EcvlSlide
from slaid.writers import Storage, _get_slide_metadata

logger = logging.getLogger(__file__)


class ZarrDirectoryStorage(Storage, _name='zarr'):
    @staticmethod
    def dump(slide: BasicSlide,
             output_path: str,
             mask_name: str = None,
             overwrite: bool = False,
             **kwargs):
        logger.info('dumping slide to zarr on path %s', output_path)
        group = zarr.open_group(output_path)
        if not group.attrs:
            group.attrs.update(_get_slide_metadata(slide))
        masks = [mask_name] if mask_name else slide.masks.keys()
        for _mask_name in masks:
            slide.masks[_mask_name].to_zarr(group, _mask_name, overwrite)

    @staticmethod
    def empty(shape, dtype):
        temp_dir = TemporaryDirectory(suffix='.zarr')
        return zarr.open(temp_dir.name, shape=shape, dtype=dtype)

    @staticmethod
    def zeros(shape, dtype):
        return zarr.zeros(shape=shape, dtype=dtype)

    @staticmethod
    def load(path: str) -> BasicSlide:
        logger.info('loading slide from zarr at path %s', path)
        group = zarr.open_group(path)
        try:
            slide = EcvlSlide(group.attrs['filename'])
        except BasicSlide.InvalidFile:
            # FIXME: workaround for cwl
            slide = EcvlSlide(os.path.basename(group.attrs['filename']))
        for name, value in group.arrays():
            try:
                logger.info('loading mask %s, %s', name, value.attrs.asdict())
                kwargs = value.attrs.asdict()
                if 'datetime' in kwargs:
                    kwargs['datetime'] = dt.fromtimestamp(kwargs['datetime'])
                slide.masks[name] = Mask(value, **kwargs)
            except Exception as ex:
                logger.error('skipping mask %s, exception: %s ', name, ex)
                raise ex
        return slide

    @staticmethod
    def mask_exists(path: str, mask: 'str') -> bool:
        if not os.path.exists(path):
            return False
        group = zarr.open_group(path)
        return mask in group.array_keys()


class ZarrZipStorage(ZarrDirectoryStorage, _name='zip'):
    @staticmethod
    def load(path: str) -> BasicSlide:
        logger.info('loading slide from zarr at path %s', path)
        storage = zarr.storage.ZipStore(path, mode='r')
        group = zarr.open_group(storage)
        try:
            slide = EcvlSlide(group.attrs['filename'])
        except BasicSlide.InvalidFile:
            # FIXME: workaround for cwl
            slide = EcvlSlide(os.path.basename(group.attrs['filename']))
        for name, value in group.arrays():
            try:
                logger.info('loading mask %s, %s', name, value.attrs.asdict())
                kwargs = value.attrs.asdict()
                if 'datetime' in kwargs:
                    kwargs['datetime'] = dt.fromtimestamp(kwargs['datetime'])
                slide.masks[name] = Mask(value, **kwargs)
            except Exception as ex:
                logger.error('skipping mask %s, exception: %s ', name, ex)
        return slide

    @staticmethod
    def dump(slide: BasicSlide,
             output_path: str,
             mask_name: str = None,
             overwrite: bool = False,
             **kwargs):
        # FIXME duplicated code
        logger.info('dumping slide to zarr on path %s', output_path)
        storage = zarr.storage.ZipStore(output_path)
        group = zarr.open_group(storage)
        if not group.attrs:
            group.attrs.update(_get_slide_metadata(slide))

        masks = [mask_name] if mask_name else slide.masks.keys()
        for _mask_name in masks:
            slide.masks[_mask_name].to_zarr(group, _mask_name, overwrite)
        storage.close()

    @staticmethod
    def mask_exists(path: str, mask: 'str') -> bool:
        if not os.path.exists(path):
            return False
        store = zarr.storage.ZipStore(path)
        group = zarr.open_group(store)
        return mask in group.array_keys()
