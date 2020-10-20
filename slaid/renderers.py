import abc
import json
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tifffile
import zarr

from slaid.classifiers.base import Patch
from slaid.commons import Mask, Slide
from slaid.commons.ecvl import Slide as EcvlSlide


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
                 rgb: bool = True):
        self.tile_size = tile_size
        self.channels = 4 if rgb else 2
        self.rgb = rgb

    def tiles(self, data: np.ndarray) -> np.ndarray:
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
                final_tile[final_tile[:, :, 0] > 0, self.channels - 1] = 255
                yield final_tile

    def render(self, array: np.ndarray, filename: str):
        with tifffile.TiffWriter(filename, bigtiff=True) as tif:
            tif.save(self.tiles(array),
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


class PatchJSONEncoder(BaseJSONEncoder):
    @property
    def target(self):
        return Patch

    def encode(self, patch: Patch):
        return {
            'slide': patch.slide.ID,
            'x': patch.x,
            'y': patch.y,
            'size': patch.size,
            'features': patch.features
        }


class NumpyArrayJSONEncoder(BaseJSONEncoder):
    @property
    def target(self):
        return np.ndarray

    def encode(self, array: np.ndarray):
        return array.tolist()


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


class SlideJSONEncoder(BaseJSONEncoder):
    @property
    def target(self):
        return Slide

    def encode(
        self,
        slide: Slide,
    ) -> Union[List, Dict]:
        dct = dict(filename=slide.filename, masks={})

        for k, v in slide.masks.items():
            dct['masks'][k] = dict(array=v.array.tolist(),
                                   extraction_level=v.extraction_level,
                                   level_downsample=v.level_downsample)
        return dct


class JSONEncoder(json.JSONEncoder):
    encoders = [
        NumpyArrayJSONEncoder(),
        SlideJSONEncoder(),
    ]

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


def to_zarr(slide: Slide, path: str):
    group = zarr.open_group(path)
    if 'slide' not in group.attrs:
        group.attrs['slide'] = slide.filename
    for name, mask in slide.masks.items():
        array = group.array(name, mask.array)
        array.attrs['level'] = mask.extraction_level
        array.attrs['downsample'] = mask.level_downsample


def from_zarr(path: str) -> Slide:
    group = zarr.open_group(path)
    slide = EcvlSlide(group.attrs['slide'])
    for name, value in group.arrays():
        slide.masks[name] = Mask(np.array(value), value.attrs['level'],
                                 value.attrs['downsample'])
    return slide
