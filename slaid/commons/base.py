import abc
import inspect
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import PIL
import shapely
from shapely.ops import cascaded_union

PATCH_SIZE = (256, 256)


def get_class(name, module):
    return dict(inspect.getmembers(sys.modules[module], inspect.isclass))[name]


class Tensor(abc.ABC):
    @abc.abstractmethod
    def getdata() -> np.ndarray:
        pass


class Image(abc.ABC):
    @abc.abstractproperty
    def dimensions(self) -> Tuple[int, int]:
        pass

    @abc.abstractmethod
    def to_array(self, PIL_FORMAT: bool = False) -> np.ndarray:
        pass

    @abc.abstractmethod
    def to_tensor(self):
        pass


Polygon = List[Tuple[int, int]]


class Mask:
    def __init__(self,
                 array: np.ndarray,
                 extraction_level: int,
                 level_downsample: int,
                 threshold: float = None):
        self.array = array
        self.extraction_level = extraction_level
        self.level_downsample = level_downsample
        self.threshold = threshold

    def to_image(self):
        return PIL.Image.fromarray(255 * self.array, 'L')

    def to_polygons(self,
                    threshold: float = None,
                    n_batch: int = 1) -> List[Polygon]:
        batch_size = self.array.shape[1] // n_batch
        polygons = []
        for batch_idx in range(n_batch):
            pos = batch_idx * batch_size
            array = self.array[:, pos:pos + batch_size]
            if threshold:
                array[array > threshold] = 1
                array[array <= threshold] = 0
                array = array.astype('uint8')

            contours, _ = cv2.findContours(array,
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)
            contours = map(np.squeeze, contours)
            contours = map(lambda x: x + (pos, 0), contours)
            pols = map(shapely.geometry.Polygon,
                       filter(lambda x: len(x) > 2, contours))
            polygons.append(cascaded_union(list(pols)))

        return cascaded_union(polygons)


class Slide(abc.ABC):
    def __init__(self, filename: str):
        if not os.path.exists(filename) or not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        self._filename = os.path.abspath(filename)
        self.masks: Dict[str, Mask] = {}

    @abc.abstractproperty
    def dimensions(self) -> Tuple[int, int]:
        pass

    @property
    def filename(self):
        return self._filename

    @abc.abstractmethod
    def read_region(self, location: Tuple[int, int],
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


def round_to_patch(coordinates, patch_size):
    res = []
    for i, c in enumerate(coordinates):
        size = patch_size[i]
        q, r = divmod(c, size)
        res.append(size * (q + round(r / size)))
    return tuple(res)
