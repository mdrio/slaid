import abc
import inspect
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime as dt
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import PIL
import tiledb
import zarr
from napari_lazy_openslide import OpenSlideStore
from napari_lazy_openslide.store import (ArgumentError, _parse_chunk_path,
                                         init_attrs)
from zarr.storage import init_array, init_group

PATCH_SIZE = (256, 256)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def get_class(name, module):
    return dict(inspect.getmembers(sys.modules[module], inspect.isclass))[name]


@dataclass
class ImageInfo(abc.ABC):
    class COLORTYPE(Enum):
        RGB = 'rgb'
        BGR = 'bgr'

    class COORD(Enum):
        XY = 'xy'
        YX = 'yx'

    class CHANNEL(Enum):
        FIRST = 'first'
        LAST = 'last'

    color_type: COLORTYPE
    coord: COORD
    channel: CHANNEL

    def __post_init__(self):
        if isinstance(self.color_type, str):
            self.color_type = ImageInfo.COLORTYPE(self.color_type)
        if isinstance(self.coord, str):
            self.coord = ImageInfo.COORD(self.coord)
        if isinstance(self.channel, str):
            self.channel = ImageInfo.CHANNEL(self.channel)


class Image(abc.ABC):
    @abc.abstractproperty
    def dimensions(self) -> Tuple[int, int]:
        pass

    @abc.abstractmethod
    def to_array(self):
        pass


@dataclass
class Polygon:
    coords: List[Tuple[int, int]]


def apply_threshold(array, threshold: float) -> np.ndarray:
    array = np.array(array)
    array[array > threshold] = 1
    array[array <= threshold] = 0
    array = array.astype('uint8')
    return array


@dataclass
class Mask:
    array: np.ndarray
    extraction_level: int
    level_downsample: int
    datetime: dt = None
    round_to_0_100: bool = False
    threshold: float = None
    model: str = None

    def _filter(self, operator: str, value: float) -> np.ndarray:
        mask = np.array(self.array)
        if self.round_to_0_100:
            mask = mask / 100
        index_patches = getattr(mask, operator)(value)
        return np.argwhere(index_patches)

    def __gt__(self, value):
        return Filter(self, self._filter('__gt__', value))

    def __ge__(self, value):
        return Filter(self, self._filter('__ge__', value))

    def __lt__(self, value):
        return Filter(self, self._filter('__lt__', value))

    def __le__(self, value):
        return Filter(self, self._filter('__le__', value))

    def __ne__(self, value):
        return Filter(self, self._filter('__ne__', value))

    def __eq__(self, other):
        if isinstance(other, float):
            return Filter(self, self._filter('__eq__', other))
        check_array = (np.array(self.array) == np.array(other.array)).all()
        return self.extraction_level == other.extraction_level \
            and self.level_downsample == other.level_downsample \
            and self.threshold == other.threshold \
            and self.model == other.model \
            and self.round_to_0_100 == other.round_to_0_100 \
            and self.datetime == other.datetime \
            and check_array

    def to_image(self, downsample: int = 1, threshold: float = None):
        array = self.array[::downsample, ::downsample]
        if self.round_to_0_100:
            array = array / 100
        if threshold:
            array = apply_threshold(array, threshold)
        return PIL.Image.fromarray((255 * array).astype('uint8'), 'L')

    def to_zarr(self, path: str, overwrite: bool = False):
        logger.info('dumping mask to zarr on path %s', path)
        name = os.path.basename(path)
        group = zarr.open_group(os.path.dirname(path))
        if overwrite and name in group:
            del group[name]
        array = group.array(name, self.array)
        for attr, value in self._get_attributes().items():
            logger.info('writing attr %s %s', attr, value)
            array.attrs[attr] = value

    def _get_attributes(self):
        attrs = {}
        attrs['extraction_level'] = self.extraction_level
        attrs['level_downsample'] = self.level_downsample
        attrs['datetime'] = self.datetime.timestamp()
        attrs['round_to_0_100'] = self.round_to_0_100
        if self.threshold:
            attrs['threshold'] = self.threshold
        if self.model:
            attrs['model'] = self.model
        return attrs

    def to_tiledb(self, path, overwrite: bool = False, ctx: tiledb.Ctx = None):
        logger.info('dumping mask to tiledb on path %s', path)
        if os.path.isdir(path) and overwrite:
            tiledb.remove(path, ctx=ctx)
        tiledb.from_numpy(path, self.array, ctx=ctx)
        self._write_meta_tiledb(path, ctx=ctx)

    def _write_meta_tiledb(self, path, ctx: tiledb.Ctx = None):
        with tiledb.open(path, 'w', ctx=ctx) as array:
            array.meta['extraction_level'] = self.extraction_level
            array.meta['level_downsample'] = self.level_downsample
            if self.threshold:
                array.meta['threshold'] = self.threshold
            if self.model:
                array.meta['model'] = self.model

    @classmethod
    def from_tiledb(cls, path, ctx: tiledb.Ctx = None):
        array = tiledb.open(path, ctx=ctx)
        return Mask(array, array.meta['extraction_level'],
                    array.meta['level_downsample'],
                    cls._get_meta(array, 'threshold'),
                    cls._get_meta(array, 'model'))

    @staticmethod
    def _get_meta(array, attr):
        try:
            res = array.meta[attr]
        except KeyError:
            res = None
        return res


@dataclass
class Filter:
    mask: Mask
    indices: np.ndarray

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        return self.indices.__getitem__(key)


def do_filter(slide: "Slide", condition: str) -> "Filter":
    operator_mapping = {
        '>': '__gt__',
        '>=': '__ge__',
        '<': '__lt__',
        '<=': '__le__',
        '==': '__eq__',
        '!=': '__ne__',
    }
    parsed = re.match(
        r"(?P<mask>\w+)\s*(?P<operator>[<>=!]+)\s*(?P<value>\d+\.*\d*)",
        condition).groupdict()
    mask = slide.masks[parsed['mask']]
    operator = operator_mapping[parsed['operator']]
    value = float(parsed['value'])
    return getattr(mask, operator)(value)


TILESIZE = 512


class BasicSlide(abc.ABC):
    class InvalidFile(Exception):
        pass

    def __init__(self, filename: str):
        self._filename = filename
        self.masks: Dict[str, Mask] = {}

    def __eq__(self, other):
        return self._filename == other.filename and self.masks == other.masks

    @abc.abstractproperty
    def dimensions(self) -> Tuple[int, int]:
        pass

    @property
    def filename(self):
        return self._filename

    @abc.abstractmethod
    def read_region(self, location: Tuple[int, int], level: int,
                    size: Tuple[int, int]) -> Image:
        pass

    @abc.abstractmethod
    def get_best_level_for_downsample(self, downsample: int):
        pass

    @abc.abstractproperty
    def level_dimensions(self):
        pass

    @abc.abstractproperty
    def level_downsamples(self):
        pass

    @property
    def level_count(self):
        return len(self.level_dimensions)


class Slide(BasicSlide):
    IMAGE_INFO = ImageInfo('bgr', 'yx', 'first')

    def __init__(self, store: "SlideStore", image_info: ImageInfo = None):
        self._store = store
        self._slide = store.slide
        self.image_info = image_info if image_info else self.IMAGE_INFO
        grp = zarr.open(store, mode="r")
        multiscales = grp.attrs["multiscales"][0]
        self._pyramid = [
            self._create_slide(d) for d in multiscales["datasets"]
        ]

    def _create_slide(self, dataset):
        return SlideArray(self._read_from_store(dataset),
                          self.IMAGE_INFO).convert(self.image_info)

    def _read_from_store(self, dataset):
        return zarr.open(store=self._store, path=dataset["path"], mode='r')
        # FIXME
        #  import dask.array as da
        #  logger.debug('dataset %s', dataset)
        #  return da.from_zarr(self._store, component=dataset["path"])

    def __getitem__(self, key) -> "SlideArray":
        return self._pyramid[key]

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self._slide.dimensions

    @property
    def filename(self):
        return self._slide.filename

    def read_region(self, location: Tuple[int, int], level: int,
                    size: Tuple[int, int]) -> Image:
        return self._slide.read_region(location, level, size)

    def get_best_level_for_downsample(self, downsample: int):
        return self._slide.get_best_level_for_downsample(downsample)

    @property
    def level_dimensions(self):
        return self._slide.level_dimensions

    @property
    def level_downsamples(self):
        return self._slide.level_downsamples


class SlideArray:
    def __init__(self, array: np.ndarray, image_info: ImageInfo = None):
        self._array = array
        self._image_info = image_info

    def __getitem__(self, key) -> "SlideArray":
        if self._is_channel_first():
            array = self._array[:, key[0], key[1]]
        else:
            array = self._array[key[0], key[1], :]
        return SlideArray(array, self._image_info)

    @property
    def array(self):
        return self._array

    def _is_channel_first(self):
        return self._image_info.channel == ImageInfo.CHANNEL.FIRST

    @property
    def size(self) -> Tuple[int, int]:
        return self._array.shape[1:] if self._is_channel_first(
        ) else self._array.shape[:2]

    def reshape(self, shape) -> "SlideArray":

        array = self._array.reshape((3, ) + shape) if self._is_channel_first(
        ) else self._array.reshape(shape + (3, ))
        return SlideArray(array, self._image_info)

    def convert(self, image_info: ImageInfo) -> "SlideArray":
        if self._image_info == image_info:
            return self

        array = np.array(self._array)
        if self._image_info.color_type != image_info.color_type:
            if self._is_channel_first():
                array = array[::-1, ...]
            else:
                array = array[..., ::-1]

        if self._image_info.channel != image_info.channel:
            if self._is_channel_first():
                array = array.transpose(1, 2, 0)
            else:
                array = array.transpose(2, 0, 1)

        return SlideArray(array, image_info)


def create_meta_store(slide: BasicSlide, tilesize: int) -> Dict[str, bytes]:
    """Creates a dict containing the zarr metadata
    for the multiscale openslide image."""
    store = dict()
    root_attrs = {
        "multiscales": [{
            "name":
            Path(slide.filename).name,
            "datasets": [{
                "path": str(i)
            } for i in range(slide.level_count)],
            "version":
            "0.1",
        }]
    }
    init_group(store)
    init_attrs(store, root_attrs)
    for i, (x, y) in enumerate(slide.level_dimensions):
        init_array(
            store,
            path=str(i),
            shape=(3, y, x),
            chunks=(3, tilesize, tilesize),
            dtype="|u1",
            compressor=None,
        )
    return store


class SlideStore(OpenSlideStore):
    def __init__(self, slide: "Slide", tilesize: int = 512):
        self._path = slide.filename
        self._slide = slide
        self._tilesize = tilesize
        self._store = create_meta_store(self._slide, tilesize)
        #  self._image_info = image_info if image_info else ImageInfo()

    @property
    def slide(self):
        return self._slide

    def __getitem__(self, key: str):
        if key in self._store:
            # key is for metadata
            return self._store[key]

        # key should now be a path to an array chunk
        # e.g '3/4.5.0' -> '<level>/<chunk_key>'
        try:
            x, y, level = _parse_chunk_path(key)
            location = self._ref_pos(x, y, level)
            size = (self._tilesize, self._tilesize)
            tile = self._slide.read_region(location, level, size)
        except ArgumentError as err:
            # Can occur if trying to read a closed slide
            raise err
        except Exception as ex:
            logger.error('ex %s', ex)
            raise KeyError(key)

        return tile.to_array().tobytes()
