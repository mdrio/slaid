import abc
import inspect
import logging
import math
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import PIL

DEFAULT_TILESIZE = 1024
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def get_class(name, module):
    return dict(inspect.getmembers(sys.modules[module], inspect.isclass))[name]


@dataclass
class ImageInfo:

    class ColorType(Enum):
        RGB = 'rgb'
        BGR = 'bgr'

    class Coord(Enum):
        XY = 'xy'
        YX = 'yx'

    class Channel(Enum):
        FIRST = 'first'
        LAST = 'last'

    class Range(Enum):
        _0_1 = '0_1'
        _1_1 = '1_1'
        _0_255 = '0_255'

    color_type: ColorType
    coord: Coord
    channel: Channel
    pixel_range: Range = Range._0_255

    _range_conversion_dict = {
        '0_255->0_1': lambda array: array / 255.,
        '0_255->1_1': lambda array: (array / 255.) * 2 - 1
    }

    @staticmethod
    def create(color_type: str,
               coord: str,
               channel: str,
               pixel_range: str = '0_255') -> "ImageInfo":
        return ImageInfo(ImageInfo.ColorType(color_type),
                         ImageInfo.Coord(coord), ImageInfo.Channel(channel),
                         ImageInfo.Range(pixel_range))

    def convert(self, array: np.ndarray,
                array_image_info: "ImageInfo") -> np.ndarray:
        if self == array_image_info:
            return array

        if self.color_type != array_image_info.color_type:
            if self.channel == ImageInfo.Channel.FIRST:
                array = array[::-1, ...]
            else:
                array = array[..., ::-1]

        if self.channel != array_image_info.channel:
            if self.channel == ImageInfo.Channel.FIRST:
                array = array.transpose(1, 2, 0)
            else:
                array = array.transpose(2, 0, 1)
        try:
            key = f'{self.pixel_range.value}->{array_image_info.pixel_range.value}'
            array = self._range_conversion_dict[key](array)
        except KeyError:
            if self.pixel_range != array_image_info.pixel_range:
                raise RuntimeError(
                    f'conversion not available from {self.pixel_range} to {array_image_info.pixel_range}'
                )

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
    slide_levels: List[Tuple[int, int]]
    round_to_0_100: bool = False
    threshold: float = None
    model: str = None
    slide: str = None
    label: str = None
    tile_size: int = 1

    def __post_init__(self):
        self.dzi_sampling_level = math.ceil(
            math.log2(max(*self.slide_levels[self.extraction_level])))

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
            and self.tile_size == other.tile_size \
            and self.slide == other.slide \
            and check_array

    def to_image(self, downsample: int = 1, threshold: float = None):
        array = self.array[::downsample, ::downsample]
        if self.round_to_0_100:
            array = array / 100
        if threshold:
            array = apply_threshold(array, threshold)
        return PIL.Image.fromarray((255 * array).astype('uint8'), 'L')

    def get_attributes(self):
        attrs = {}
        attrs['extraction_level'] = self.extraction_level
        attrs['dzi_sampling_level'] = self.dzi_sampling_level
        attrs['level_downsample'] = self.level_downsample
        attrs['round_to_0_100'] = self.round_to_0_100
        attrs['slide_levels'] = self.slide_levels
        attrs['tile_size'] = self.tile_size
        if self.threshold:
            attrs['threshold'] = self.threshold
        if self.model:
            attrs['model'] = self.model
        return attrs


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
        res = np.zeros(size, dtype='bool')
        for x in range(self._array.shape[0]):
            for y in range(self._array.shape[1]):
                res[x * scale[0]:x * scale[0] + scale[0],
                    y * scale[1]:y * scale[1] + scale[1]] = self._array[x, y]
        self._array = res


class BasicSlide(abc.ABC):
    IMAGE_INFO = ImageInfo.create('bgr', 'yx', 'first')

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

    def __init__(self, slide_reader: BasicSlide):
        self._slide_reader = slide_reader

    @property
    def masks(self):
        return self._slide_reader.masks

    @property
    def image_info(self):
        return self._slide_reader.IMAGE_INFO

    def __getitem__(self, key) -> "SlideArray":
        return BasicSlideArray(self._slide_reader, key, self.image_info)

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self._slide_reader.dimensions

    @property
    def filename(self):
        return self._slide_reader.filename

    def read_region(self, location: Tuple[int, int], level: int,
                    size: Tuple[int, int]) -> Image:
        return self._slide_reader.read_region(location, level, size)

    def get_best_level_for_downsample(self, downsample: int):
        return self._slide_reader.get_best_level_for_downsample(downsample)

    @property
    def level_dimensions(self):
        return self._slide_reader.level_dimensions

    @property
    def level_downsamples(self):
        return self._slide_reader.level_downsamples


class SlideArray(abc.ABC):

    @abc.abstractmethod
    def __getitem__(self, key) -> "SlideArray":
        ...

    @abc.abstractproperty
    def array(self):
        ...

    @abc.abstractmethod
    def convert(self, image_info: ImageInfo) -> "SlideArray":
        ...

    @abc.abstractproperty
    def size(self) -> Tuple[int, int]:
        ...


class BasicSlideArray(SlideArray):

    def __init__(self, slide: BasicSlide, level: int, image_info: ImageInfo):
        self._slide = slide
        self._level = level
        self.image_info = image_info
        self._array: np.ndarray = None

    @property
    def array(self):
        if self._array is not None:
            return self._array
        logger.warn('reading the whole slide...')

        image = self._slide.read_region(
            (0, 0), self._level, self._slide.level_dimensions[self._level])
        self._array = image.to_array()
        return self._array

    def __getitem__(self, key: Tuple[slice, slice]) -> "SlideArray":

        def get_slice_stop(slice_value: int, limit: int):
            return min(slice_value,
                       limit) if slice_value is not None else limit

        if self._array is not None:
            array = self._array[:, key[0], key[1]] if self._is_channel_first(
            ) else self._array[key[0], key[1], :]
        else:
            slice_x = key[1]
            slice_y = key[0]

            slice_x_start = slice_x.start or 0
            slice_x_stop = get_slice_stop(
                slice_x.stop, self._slide.level_dimensions[self._level][0])

            slice_y_start = slice_y.start or 0
            slice_y_stop = get_slice_stop(
                slice_y.stop, self._slide.level_dimensions[self._level][1])

            slice_x = slice(slice_x_start, slice_x_stop)
            slice_y = slice(slice_y_start, slice_y_stop)

            location = (slice_x.start, slice_y.start)
            location_at_level_0 = tuple([
                int(c * self._slide.level_downsamples[self._level])
                for c in location
            ])
            size = (slice_x.stop - slice_x.start, slice_y.stop - slice_y.start)
            array = self._slide.read_region(location_at_level_0, self._level,
                                            size).to_array()
        slide_array = self._clone(array)
        return slide_array

    def convert(self, image_info: ImageInfo) -> SlideArray:
        array = self.image_info.convert(self.array, image_info)
        return self._clone(array, image_info)

    @property
    def size(self) -> Tuple[int, int]:
        if self._array is not None:
            return self._array.shape[1:] if self._is_channel_first(
            ) else self._array.shape[:2]
        return self._slide.level_dimensions[self._level][::-1]

    def _clone(self,
               array: np.ndarray = None,
               image_info: ImageInfo = None) -> SlideArray:
        slide_array = BasicSlideArray(self._slide, self._level, image_info
                                      or self.image_info)
        slide_array._array = array
        return slide_array

    def _is_channel_first(self):
        return self.image_info.channel == ImageInfo.Channel.FIRST


class ArrayFactory(abc.ABC):

    @abc.abstractmethod
    def empty(self, shape: Tuple[int, int], dtype: str):
        ...

    @abc.abstractmethod
    def zeros(self, shape: Tuple[int, int], dtype: str):
        ...


class NumpyArrayFactory(ArrayFactory):

    def empty(self, shape: Tuple[int, int], dtype: str):
        return np.empty(shape, dtype=dtype)

    def zeros(self, shape: Tuple[int, int], dtype: str):
        return np.zeros(shape, dtype=dtype)
