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
from napari_lazy_openslide.store import ArgumentError, init_attrs
from skimage.util import view_as_blocks
from zarr.storage import init_array, init_group

_TILESIZE = 2048
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

    def convert(self, array: np.ndarray,
                array_image_info: "ImageInfo") -> np.ndarray:
        if self == array_image_info:
            return array

        if self.color_type != array_image_info.color_type:
            if self.channel == ImageInfo.CHANNEL.FIRST:
                array = array[::-1, ...]
            else:
                array = array[..., ::-1]

        if self.channel != array_image_info.channel:
            if self.channel == ImageInfo.CHANNEL.FIRST:
                array = array.transpose(1, 2, 0)
            else:
                array = array.transpose(2, 0, 1)

        return array


class Image(abc.ABC):
    @abc.abstractproperty
    def dimensions(self) -> Tuple[int, int]:
        pass

    @abc.abstractmethod
    def to_array(self, image_info: ImageInfo = None):
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
        return getattr(mask, operator)(value)

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


class Filter:
    def __init__(self, mask: Mask, array: np.ndarray):
        self._mask = mask

        self._indices = None
        self._array = array

    @property
    def array(self):
        return self._array

    def __iter__(self):
        return iter(self._array)

    #  def __len__(self):
    #      return len(self.indices)

    def __getitem__(self, key):
        return self._array.__getitem__(key)

    def rescale(self, size: int):
        if size < self._array.shape:
            raise NotImplementedError('size %s < %s', size, self._array.shape)
        if size == self._array.shape:
            return self._array
        scale = (size[0] // int(self._array.shape[0]),
                 size[1] // int(self._array.shape[1]))
        print(scale)
        res = np.zeros(size, dtype='bool')
        for x in range(self._array.shape[0]):
            for y in range(self._array.shape[1]):
                res[x * scale[0]:x * scale[0] + scale[0],
                    y * scale[1]:y * scale[1] + scale[1]] = self._array[x, y]
        self._array = res


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


class BasicSlide(abc.ABC):
    IMAGE_INFO = ImageInfo('bgr', 'yx', 'first')

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
    def __init__(self, store: "SlideStore", image_info: ImageInfo = None):
        self._store = store
        self._slide = store.slide
        self.image_info = image_info if image_info else self.IMAGE_INFO
        grp = zarr.open(store, mode="r")
        multiscales = grp.attrs["multiscales"][0]
        self._pyramid = [
            self._create_slide(d) for d in multiscales["datasets"]
        ]

    @property
    def masks(self):
        return self._slide.masks

    def _create_slide(self, dataset):
        return SlideArray(self._read_from_store(dataset),
                          self._slide.IMAGE_INFO).convert(self.image_info)

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
            array = self._array[:, key[0], key[1]] if isinstance(
                key, tuple) else self._array[:, key]
        else:
            array = self._array[key[0], key[1], :] if isinstance(
                key, tuple) else self._array[key, :]
        return self.__class__(array, self._image_info)

    @property
    def array(self):
        return np.array(self._array)

    def _is_channel_first(self):
        return self._image_info.channel == ImageInfo.CHANNEL.FIRST

    @property
    def size(self) -> Tuple[int, int]:
        return self._array.shape[1:] if self._is_channel_first(
        ) else self._array.shape[:2]

    def reshape(self, shape) -> "SlideArray":

        array = self._array.reshape((3, ) + shape) if self._is_channel_first(
        ) else self._array.reshape(shape + (3, ))
        return self.__class__(array, self._image_info)

    def convert(self, image_info: ImageInfo) -> "SlideArray":
        array = self._image_info.convert(self.array, image_info)
        return self.__class__(array, image_info)

    def apply_filter(self, filter_: Filter) -> "SlideArray":
        if self._is_channel_first():
            array = self.array[:, filter_.array]
        else:
            array = self.array[filter_.array, :]

        return self.__class__(array, self._image_info)

    def get_blocks(self, block_shape: Tuple[int,
                                            int]) -> Tuple[np.ndarray, bool]:
        if self._is_channel_first():
            block_shape = (3, ) + block_shape
            channel_first = True
        else:

            block_shape = block_shape + (3, )
            channel_first = False
        array = view_as_blocks(self.array, block_shape)
        return array, channel_first


def _create_metastore(slide: BasicSlide, tilesize: int) -> Dict[str, bytes]:
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
            shape=(3, y, x) if slide.IMAGE_INFO.channel
            == ImageInfo.CHANNEL.FIRST else (y, x, 3),
            chunks=(3, tilesize, tilesize) if slide.IMAGE_INFO.channel
            == ImageInfo.CHANNEL.FIRST else (tilesize, tilesize, 3),
            dtype="|u1",
            compressor=None,
        )
    return store


class SlideStore(OpenSlideStore):
    def __init__(self, slide: "Slide", tilesize: int = _TILESIZE):
        self._path = slide.filename
        self._slide = slide
        self._tilesize = tilesize
        self._store = _create_metastore(self._slide, tilesize)
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
            x, y, level = self._parse_chunk_path(key)
            location = self._ref_pos(x, y, level)
            size = (self._tilesize, self._tilesize)
            logger.debug(
                'key %s, x %s, y %s, location %s, size %s, dimension %s', key,
                x, y, location, size, self._slide.level_dimensions[level])
            tile = self._slide.read_region(location, level, size)
        except ArgumentError as err:
            # Can occur if trying to read a closed slide
            raise err
        except Exception as ex:
            logger.error('ex %s', ex)
            raise KeyError(key)

        return tile.to_array().tobytes()

    def _parse_chunk_path(self, path: str):
        """Returns x,y chunk coords and pyramid level from string key"""
        level, ckey = path.split("/")
        _, y, x = map(int, ckey.split("."))
        if self._slide.IMAGE_INFO.channel == ImageInfo.CHANNEL.LAST:
            y, x, _ = _, y, x
        return x, y, int(level)


class BaseSlideFactory(abc.ABC):
    def __init__(self,
                 filename: str,
                 basic_slide_module: str,
                 slide_module: str,
                 tilesize: int = _TILESIZE,
                 image_info: ImageInfo = None):
        self._filename = filename.rstrip('/')
        self._basic_slide_module = basic_slide_module
        self._slide_module = slide_module
        self._tilesize = tilesize
        self._image_info = image_info

    @abc.abstractmethod
    def get_slide(self) -> Slide:
        pass
