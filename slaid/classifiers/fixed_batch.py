import logging
from dataclasses import dataclass
from datetime import datetime as dt
from functools import partial
from typing import Callable
from slaid.classifiers.base import Classifier, append_array

import numpy as np

from slaid.commons import Filter, Mask, NapariSlide
from slaid.commons.base import ImageInfo
from slaid.models import Model

logger = logging.getLogger()


@dataclass
class BatchIterator:
    batch_size: int
    channel_first: bool

    def __post_init__(self):
        self._buffer: np.ndarray = np.empty((3,
                                             0) if self.channel_first else (0,
                                                                            3))

    @property
    def buffer(self):
        return self._buffer

    def iter(self) -> np.ndarray:
        max_index = self._buffer_size() - self._buffer_size() % self.batch_size
        for i in range(0, max_index, self.batch_size):
            batch = self._buffer[:, i:i + self.
                                 batch_size] if self.channel_first else self._buffer[
                                     i:i + self.batch_size, :]
            yield batch
        self._buffer = self._buffer[:,
                                    max_index:] if self.channel_first else self._buffer[
                                        max_index:, :]

    def append(self, array: np.ndarray):
        self._buffer = np.concatenate([self._buffer, array])

    def _buffer_size(self):
        return self._buffer.shape[
            1] if self.channel_first else self._buffer.shape[0]


class FilteredClassifier(Classifier):

    def __init__(
        self,
        model: "Model",
        feature: str,
        _filter: Filter,
        array_factory: Callable = np,
    ):
        super().__init__(model, feature, array_factory)
        self._filter = _filter


class PixelClassifier(Classifier):

    def __init__(self,
                 model: "Model",
                 feature: str,
                 array_factory: Callable = np,
                 chunk_size: int = None):
        super().__init__(model, feature, array_factory)
        self.chunk_size = chunk_size

    def classify(self,
                 slide: NapariSlide,
                 level,
                 threshold: float = None,
                 batch_size: int = 8,
                 round_to_0_100: bool = True) -> Mask:

        slide_array = slide[level]
        row_size = self.chunk_size if self.chunk_size else slide_array.size[0]
        dtype = 'uint8' if threshold or round_to_0_100 else 'float32'

        res = self._array_factory.empty((0, ), dtype=dtype)

        channel_first = self.model.image_info.CHANNEL == ImageInfo.CHANNEL.FIRST
        batch_iterator = BatchIterator(batch_size, channel_first)
        for row_idx in range(0, slide_array.size[0], row_size):
            row = slide_array[row_idx:row_idx + row_size, :].convert(
                self.model.image_info).array
            row = row.reshape(3, -1) if channel_first else row.reshape(-1, 3)
            batch_iterator.append(row)
            predictions = []
            for batch in batch_iterator.iter():
                predictions.append(self._predict(batch))

            res = append_array(res, np.concatenate(predictions),
                               0) if predictions else res

        res = append_array(res, self._predict(batch_iterator.buffer), 0)

        res = res.reshape(slide_array.size)
        res = self._threshold(res, threshold)
        res = self._round_to_0_100(res, round_to_0_100)

        return self._get_mask(slide,
                              res, level, slide.level_downsamples[level],
                              dt.now(), round_to_0_100)


class FilteredPatchClassifier(FilteredClassifier):

    def classify(self,
                 slide: NapariSlide,
                 level: int,
                 threshold: float = None,
                 batch_size: int = 8,
                 round_to_0_100: bool = True) -> Mask:
        if not self._patch_size:
            raise RuntimeError(f'invalid patch size {self._patch_size}')

        slide_array = slide[level]

        filter_array = self._filter.array
        patch_coords = np.argwhere(filter_array) * self._patch_size
        patch_coords = list(
            filter(partial(self._remove_borders, slide_array),
                   iter(patch_coords)))

        n_patches = len(patch_coords)
        dtype = 'uint8' if threshold or round_to_0_100 else 'float32'
        res = self._array_factory.zeros(
            (slide_array.size[0] // self._patch_size[0],
             slide_array.size[1] // self._patch_size[1]),
            dtype=dtype)

        predictions = np.empty(n_patches)
        patches = []
        for patch_coord in patch_coords:
            x, y = patch_coord

            patch = slide_array[x:x + self._patch_size[0],
                                y:y + self._patch_size[1]].convert(
                                    self.model.image_info).array
            patches.append(patch)

        for index in range(0, n_patches, batch_size):

            to_predict = np.array(patches[index:index + batch_size])
            prediction = self._predict(to_predict)
            predictions[index:index + batch_size] = prediction

        res[filter_array] = predictions
        res = self._threshold(res, threshold)
        res = self._round_to_0_100(res, round_to_0_100)

        return self._get_mask(slide,
                              res, level, slide.level_downsamples[level],
                              dt.now(), round_to_0_100)

    def _remove_borders(self, slide_array: np.ndarray,
                        coord: np.ndarray) -> bool:
        return coord[0] <= (slide_array.size[0] -
                            self._patch_size[0]) and coord[1] <= (
                                slide_array.size[1] - self._patch_size[1])


class FilteredPixelClassifier(FilteredClassifier):

    def classify(self,
                 slide: NapariSlide,
                 threshold: float = None,
                 level: int = 2,
                 batch_size: int = 4096,
                 round_to_0_100: bool = True) -> Mask:
        if self._patch_size:
            raise RuntimeError(
                f'Not expecting patch size, found {self._patch_size}')

        slide_array = slide[level]
        filter_array = self._filter.array
        zoom_factor = (
            slide_array.size[0] // filter_array.shape[0],
            slide_array.size[1] // filter_array.shape[1],
        )
        tile_size = zoom_factor
        patch_coords = np.argwhere(filter_array) * tile_size
        #  patch_coords = list(
        #      filter(partial(self._remove_borders, slide_array),
        #             iter(patch_coords)))
        #
        n_patches = len(patch_coords)
        predictions = np.empty(n_patches)
        patches = []
        for patch_coord in patch_coords:
            x, y = patch_coord

            patch = slide_array[x:x + tile_size[0],
                                y:y + tile_size[1]].convert(
                                    self.model.image_info).array
            patches.append(patch.reshape(-1, 3))

        to_predict = np.concatenate(patches) if patches else np.empty(
            (0, 3), dtype='uint8')
        predictions = self._predict(to_predict)

        dtype = 'uint8' if threshold or round_to_0_100 else 'float32'
        res = self._array_factory.zeros(slide_array.size, dtype=dtype)
        patch_area = tile_size[0] * tile_size[1]

        for i in range(n_patches):
            patch_coord = patch_coords[i]
            x, y = patch_coord
            i = i * patch_area
            patch = predictions[i:i + patch_area].reshape(tile_size)
            patch = self._threshold(patch, threshold)
            patch = self._round_to_0_100(patch, round_to_0_100)
            res[x:x + tile_size[0], y:y + tile_size[1]] = patch

        return self._get_mask(slide,
                              res, level, slide.level_downsamples[level],
                              dt.now(), round_to_0_100)
