import abc
import inspect
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import PIL
import shapely
import tiledb
from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon

PATCH_SIZE = (256, 256)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


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


@dataclass
class Polygon:
    coords: List[Tuple[int, int]]


def apply_threshold(array, threshold: float) -> np.ndarray:
    array = np.array(array)
    array[array > threshold] = 1
    array[array <= threshold] = 0
    array = array.astype('uint8')
    return array


def mask_to_polygons(mask, epsilon=10., min_area=10.):
    """
    https://stackoverflow.com/questions/60971260/how-to-transform-contours-obtained-from-opencv-to-shp-file-polygons

    the original source of these helpers was a Kaggle
    post by Konstantin Lopuhin here - you'll need
    to be logged into Kaggle to see it
    """
    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)
    if not contours:
        return ShapelyMultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = ShapelyPolygon(shell=cnt[:, 0, :],
                                  holes=[
                                      c[:, 0, :]
                                      for c in cnt_children.get(idx, [])
                                      if cv2.contourArea(c) >= min_area
                                  ])
            all_polygons.append(poly)
    all_polygons = ShapelyMultiPolygon(all_polygons)

    return all_polygons


class Mask:
    def __init__(self,
                 array: np.ndarray,
                 extraction_level: int,
                 level_downsample: int,
                 threshold: float = None,
                 model: str = None):
        self.array = array
        self.extraction_level = extraction_level
        self.level_downsample = level_downsample
        self.threshold = threshold
        self.model = model

    def to_image(self, downsample: int = 1, threshold: float = None):
        array = self.array[::downsample, ::downsample]
        if threshold:
            array = apply_threshold(array, threshold)
        return PIL.Image.fromarray(255 * array, 'L')

    def to_polygons(self,
                    threshold: float = None,
                    downsample: int = 1,
                    n_batch: int = 1) -> List[Polygon]:
        array = np.array(self.array[::downsample, ::downsample])
        if threshold:
            array[array > threshold] = 1
            array[array <= threshold] = 0
            array = array.astype('uint8')
            polygons = mask_to_polygons(array)
        return [
            Polygon(
                list(
                    shapely.affinity.scale(p,
                                           downsample,
                                           downsample,
                                           origin=(0, 0)).exterior.coords))
            for p in polygons
        ]

    def to_tiledb(self, path, **kwargs):
        tiledb.from_numpy(path, self.array)
        print(f'kwargs {kwargs}')
        self._write_meta_tiledb(path, **kwargs)

    def _write_meta_tiledb(self, path, **kwargs):
        with tiledb.open(path, 'w', **kwargs) as array:
            array.meta['extraction_level'] = self.extraction_level
            array.meta['level_downsample'] = self.level_downsample
            if self.threshold:
                array.meta['threshold'] = self.threshold
            if self.model:
                array.meta['model'] = self.model

    #  def from_tiledb(path, **kwargs)

    def to_zarr(self, path, **kwargs):
        pass


class Slide(abc.ABC):
    def __init__(self, filename: str):
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
