import abc
import json
import logging
import os
from typing import Any, Tuple, Union

import numpy as np
import tifffile
import tiledb
import zarr

from slaid.commons import Mask, Slide
from slaid.commons.base import Polygon
from slaid.commons.ecvl import Slide as EcvlSlide

logger = logging.getLogger(__file__)


class Renderer(abc.ABC):
    @abc.abstractmethod
    def render(
        self,
        array: np.ndarray,
        filename: str,
    ):
        pass


class TiffRenderer(Renderer):
    def __init__(self,
                 tile_size: Tuple[int, int] = (256, 256),
                 rgb: bool = True,
                 bigtiff=True):
        self.tile_size = tile_size
        self.channels = 4 if rgb else 2
        self.rgb = rgb
        self.bigtiff = bigtiff

    def _tiles(self, data: np.ndarray) -> np.ndarray:
        for y in range(0, data.shape[0], self.tile_size[0]):
            for x in range(0, data.shape[1], self.tile_size[1]):
                tile = data[y:y + self.tile_size[0], x:x + self.tile_size[1]]
                if tile.shape[:2] != self.tile_size:
                    pad = (
                        (0, self.tile_size[0] - tile.shape[0]),
                        (0, self.tile_size[1] - tile.shape[1]),
                    )
                    tile = np.pad(tile, pad, 'constant')
                final_tile = np.zeros(
                    (tile.shape[0], tile.shape[1], self.channels),
                    dtype='uint8')

                final_tile[:, :, 0] = tile * 255
                final_tile[final_tile[:, :, 0] > 255 / 10,
                           self.channels - 1] = 255
                yield final_tile

    def render(self, array: np.ndarray, filename: str):
        with tifffile.TiffWriter(filename, bigtiff=self.bigtiff) as tif:
            tif.save(self._tiles(array),
                     dtype='uint8',
                     shape=(array.shape[0], array.shape[1], self.channels),
                     tile=self.tile_size,
                     photometric='rgb' if self.rgb else 'minisblack',
                     extrasamples=('ASSOCALPHA', ))


class BaseJSONEncoder(abc.ABC):
    @abc.abstractproperty
    def target(self):
        pass

    def encode(self, obj: Any):
        pass


class NumpyArrayJSONEncoder(BaseJSONEncoder):
    @property
    def target(self):
        return np.ndarray

    def encode(self, array: np.ndarray):
        return array.tolist()


class PolygonJSONEncoder(BaseJSONEncoder):
    @property
    def target(self):
        return Polygon

    def encode(self, obj: Polygon):
        return obj.coords


class Int64JSONEncoder(BaseJSONEncoder):
    @property
    def target(self):
        return np.int64

    def encode(self, int_: np.int64):
        return int(int_)


# from https://github.com/hmallen/numpyencoder
def convert_numpy_types(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):

        return int(obj)

    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)

    elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return {'real': obj.real, 'imag': obj.imag}

    elif isinstance(obj, (np.ndarray, )):
        return obj.tolist()

    elif isinstance(obj, (np.bool_)):
        return bool(obj)

    elif isinstance(obj, (np.void)):
        return None
    return obj


class JSONEncoder(json.JSONEncoder):
    encoders = [NumpyArrayJSONEncoder(), PolygonJSONEncoder()]

    def default(self, obj):
        encoded = None
        for encoder in self.encoders:
            if isinstance(obj, encoder.target):
                encoded = encoder.encode(obj)
                break
        if encoded is None:
            encoded = super().default(obj)
        return encoded


class VectorialRenderer(Renderer):
    def render(self,
               slide: Slide,
               filename: str,
               one_file_per_patch: bool = False):
        if one_file_per_patch:
            raise NotImplementedError()
        with open(filename, 'w') as json_file:
            json.dump(slide.patches, json_file, cls=JSONEncoder)


def to_json(obj: Any, filename: str = None) -> Union[str, None]:
    if filename is not None:
        with open(filename, 'w') as f:
            json.dump(obj, f, cls=JSONEncoder)
    else:
        return json.dumps(obj, cls=JSONEncoder)


def to_zarr(slide: Slide,
            path: str,
            mask: str = None,
            overwrite: bool = False,
            **kwargs) -> str:
    output_path = os.path.join(path,
                               f'{os.path.basename(slide.filename)}.zarr')
    logger.info('dumping slide to zarr on path %s', output_path)
    group = zarr.open_group(output_path)
    if 'slide' not in group.attrs:
        group.attrs['slide'] = slide.filename
    _dump_masks(output_path, slide, overwrite, 'to_zarr', mask, **kwargs)
    return output_path


def _dump_masks(path: str,
                slide: Slide,
                overwrite: bool,
                func: str,
                only_mask: str = None,
                **kwargs):
    masks = {only_mask: slide.masks[only_mask]} if only_mask else slide.masks
    for name, mask_ in masks.items():
        getattr(mask_, func)(os.path.join(path, name), overwrite, **kwargs)


def from_zarr(path: str) -> Slide:
    logger.info('loading slide from zarr at path %s', path)
    group = zarr.open_group(path)
    slide = EcvlSlide(group.attrs['slide'])
    for name, value in group.arrays():
        logger.info('loading mask %s', name)
        slide.masks[name] = Mask(value, **value.attrs)
    return slide


def to_tiledb(slide: Slide,
              path: str,
              overwrite: bool = False,
              mask: str = False,
              ctx: tiledb.Ctx = None) -> str:
    if not os.path.isdir(path):
        logger.info('creating tiledb group at path %s', path)
        tiledb.group_create(path, ctx=ctx)
    output_path = os.path.join(path,
                               f'{os.path.basename(slide.filename)}.tiledb')
    os.makedirs(output_path, exist_ok=True)
    slide_filename = os.path.join(output_path, '.slide')

    with open(slide_filename, 'w') as f:
        logger.debug('writing slide file on %s', slide_filename)
        f.write(slide.filename)

    _dump_masks(output_path, slide, overwrite, 'to_tiledb', mask, ctx=ctx)
    return output_path


def from_tiledb(path: str, ctx: tiledb.Ctx = None) -> Slide:
    with open(os.path.join(path, '.slide'), 'r') as f:
        filename = f.read()
    slide = EcvlSlide(filename)
    masks = []
    tiledb.ls(path, lambda obj_path, obj_type: masks.append(obj_path), ctx=ctx)
    for name in masks:
        slide.masks[os.path.basename(name)] = Mask.from_tiledb(name, ctx=ctx)
    return slide
