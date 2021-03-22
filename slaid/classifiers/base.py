import abc
import logging
from dataclasses import dataclass
from datetime import datetime as dt
from typing import Callable, Tuple

import numpy as np
from progress.bar import Bar

from slaid.commons import BasicSlide, Mask, Slide
from slaid.commons.base import Filter, ImageInfo
from slaid.models import Model

logger = logging.getLogger('classify')
fh = logging.FileHandler('/tmp/base-classifier.log')
logger.addHandler(fh)


class Classifier(abc.ABC):
    @abc.abstractmethod
    def classify(self,
                 slide: BasicSlide,
                 filter_=None,
                 threshold: float = None,
                 level: int = 2,
                 n_batch: int = 1,
                 round_to_0_100: bool = True) -> Mask:
        pass


class BasicClassifier(Classifier):
    MASK_CLASS = Mask

    def __init__(self, model: "Model", feature: str):
        self._model = model
        self.feature = feature
        try:
            self._patch_size = self._model.patch_size
            self._image_info = model.image_info
        except AttributeError as ex:
            logger.error(ex)
            self._patch_size = None
            self._image_info = ImageInfo(ImageInfo.COLORTYPE('rgb'),
                                         ImageInfo.COORD('yx'),
                                         ImageInfo.CHANNEL('last'))

    @property
    def model(self):
        return self._model

    def classify(self,
                 slide: Slide,
                 filter_=None,
                 threshold: float = None,
                 level: int = 2,
                 n_batch: int = 1,
                 round_to_0_100: bool = True,
                 n_patch=25) -> Mask:

        logger.info('classify: %s, %s, %s, %s, %s, %s', slide.filename,
                    filter_, threshold, level, n_batch, round_to_0_100)
        #  batches = self._get_batch_iterator(slide, level, n_batch,
        #                                     self._color_type, self._coords,
        #                                     self._channel)
        if self._patch_size:
            predictions = self._classify_patches(slide, self._patch_size,
                                                 level, filter_, threshold,
                                                 n_patch, round_to_0_100)
        else:
            predictions = self._classify_batches(slide, level, n_batch,
                                                 filter_)
        predictions = self._threshold(predictions, threshold)
        predictions = self._round_to_0_100(predictions, round_to_0_100)
        return self._get_mask(predictions, level,
                              slide.level_downsamples[level], dt.now(),
                              round_to_0_100)

    @staticmethod
    def _round_to_0_100(array: np.ndarray, round_: bool) -> np.ndarray:
        return (array * 100).astype('uint8') if round_ else array

    @staticmethod
    def _threshold(array: np.ndarray, threshold: float) -> np.ndarray:
        if threshold is not None:
            array[array >= threshold] = 1
            array[array < threshold] = 0
            return array.astype('uint8')
        return array

    def _get_mask(self, array, level, downsample, datetime, round_to_0_100):
        return self.MASK_CLASS(array,
                               level,
                               downsample,
                               datetime,
                               round_to_0_100,
                               model=str(self._model))

    def _classify_patches(self,
                          slide: BasicSlide,
                          patch_size,
                          level,
                          filter_: Filter,
                          threshold,
                          n_patch: int = 25,
                          round_to_0_100: bool = True) -> Mask:
        dimensions = slide.level_dimensions[level][::-1]
        dtype = 'uint8' if threshold or round_to_0_100 else 'float32'
        res = np.zeros(
            (dimensions[0] // patch_size[0], dimensions[1] // patch_size[1]),
            dtype=dtype)

        patch_indexes = filter_ if filter_ is not None else np.ndindex(
            dimensions[0] // patch_size[0], dimensions[1] // patch_size[1])
        patches_to_predict = [
            Patch(slide, p[0], p[1], level, patch_size, self._image_info)
            for p in patch_indexes
        ]

        # adding fake patches, workaround for
        # https://github.com/deephealthproject/eddl/issues/236
        #  for _ in range(patch_to_add):
        #      patches_to_predict.append(patches_to_predict[0])

        predictions = []
        with Bar('Predictions', max=len(patches_to_predict) // n_patch
                 or 1) as predict_bar:
            for i in range(0, len(patches_to_predict), n_patch):
                patches = patches_to_predict[i:i + n_patch]
                predictions.append(
                    self._classify_array(
                        np.stack([p.array() for p in patches]), threshold,
                        round_to_0_100))
                predict_bar.next()
        if predictions:
            predictions = np.concatenate(predictions)
        for i, p in enumerate(predictions):
            patch = patches_to_predict[i]
            res[patch.row, patch.column] = p
        return res

    def _classify_batches(self, slide, level, n_batch, filter_):
        slide_array = self._get_slide_array(slide, level)
        if filter_ is None:
            return self._classify_batches_no_filter(slide_array, level,
                                                    n_batch)

        else:
            return self._classify_batches_with_filter(slide, slide_array,
                                                      level, n_batch, filter_)

    def _get_slide_array(self, slide, level):
        return slide[level].convert(self.model.image_info)

    def _classify_batches_with_filter(self, slide, slide_array, level, n_batch,
                                      filter_):
        scale_factor = round(
            slide.level_downsamples[filter_.mask.extraction_level]) // round(
                slide.level_downsamples[level])

        res = np.zeros(slide_array.size, dtype='float32')
        for pixel in filter_:
            pixel = pixel * scale_factor
            area = slide_array[pixel[0]:pixel[0] + scale_factor,
                               pixel[1]:pixel[1] + scale_factor]

            n_px = area.size[0] * area.size[1]
            prediction = self._predict(area.reshape((n_px, )))
            res[pixel[0]:pixel[0] + scale_factor,
                pixel[1]:pixel[1] + scale_factor] = prediction.reshape(
                    (scale_factor, scale_factor))
        return res

    def _classify_batches_no_filter(self, slide_array, level, n_batch):
        predictions = []
        step = slide_array.size[0] // n_batch
        logger.debug('n_batch %s, step %s', n_batch, step)
        with Bar('batches', max=n_batch) as bar:
            for i in range(0, slide_array.size[0], step):
                area = slide_array[i:i + step, :]
                n_px = area.size[0] * area.size[1]
                area_reshaped = area.reshape((n_px, ))

                prediction = self._predict(area_reshaped)
                prediction = prediction.reshape(area.size[0], area.size[1])
                predictions.append(prediction)
                bar.next()
        return self._concatenate(predictions, 0)

    def _predict(self, area):
        return self.model.predict(area.array)

    def _classify_batch(self, batch, threshold, round_to_0_100):
        logger.debug('classify batch %s', batch)
        # FIXME
        filter_ = None
        array = batch.array()
        logger.debug('get array')
        orig_shape = batch.shape
        if filter_ is not None:
            indexes_pixel_to_process = filter_.filter(batch)
            array = array[indexes_pixel_to_process]
        else:
            logger.debug('flattening batch')
            array = batch.flatten()

        logger.debug('start predictions')
        prediction = self._classify_array(array, threshold, round_to_0_100)
        #      self._classify_array(_, threshold, round_to_0_100)
        #  prediction = np.concatenate([
        #      self._classify_array(_, threshold, round_to_0_100)
        #      for _ in np.array_split(array, 10000)
        #  ])
        #  logger.debug('end predictions')
        #      self._classify_array(_, threshold, round_to_0_100)
        if filter_ is not None:
            res = np.zeros(orig_shape[:2], dtype=prediction.dtype)
            res[indexes_pixel_to_process] = prediction
        else:
            res = prediction.reshape(orig_shape[:2])
        return res

    def _classify_array(self, array, threshold, round_to_0_100) -> np.ndarray:
        logger.debug('classify array')
        prediction = self.model.predict(array)
        if round_to_0_100:
            prediction = prediction * 100
            return prediction.astype('uint8')
        if threshold:
            prediction[prediction >= threshold] = 1
            prediction[prediction < threshold] = 0
            return prediction.astype('uint8')

        return prediction.astype('float32')

    @staticmethod
    def _get_batch(slide, start, size, level):
        image = slide.read_region(start[::-1], level, size[::-1])
        array = image.to_array(True)
        return Batch(start, size, array, slide.level_downsamples[level])

    @staticmethod
    def _get_zeros(size, dtype):
        return np.zeros(size, dtype)

    @staticmethod
    def _concatenate(seq, axis):
        return np.concatenate(seq, axis)

    @staticmethod
    def _reshape(array, shape):
        return array.reshape(shape)

    def _flat_array(self, array: np.ndarray, shape: Tuple[int,
                                                          int]) -> np.ndarray:
        n_px = shape[0] * shape[1]
        array = array[:, :, :3].reshape(n_px, 3)
        return array


@dataclass
class ImageArea:
    slide: BasicSlide
    row: int
    column: int
    level: int
    size: Tuple[int, int]
    image_info: ImageInfo = ImageInfo(ImageInfo.COLORTYPE.BGR,
                                      ImageInfo.COORD.YX,
                                      ImageInfo.CHANNEL.FIRST)

    def __post_init__(self):
        self._array = None

    @property
    def top_level_location(self):
        raise NotImplementedError

    def array(self):
        logger.debug('reading array of %s', self)
        if self._array is None:
            image = self.slide.read_region(self.top_level_location[::-1],
                                           self.level, self.size[::-1])
            self._array = image.to_array(self.image_info)
        return self._array

    @property
    def shape(self):
        return self.array().shape[:2] if self.image_info.channel == ImageInfo.CHANNEL.LAST \
            else self.array().shape[1:]

    def flatten(self):
        n_px = self.shape[0] * self.shape[1]
        if self.image_info.channel == ImageInfo.CHANNEL.FIRST:
            array = self.array().transpose(1, 2, 0)
        else:
            array = self.array()
        array = array[:, :, :3].reshape(n_px, 3)
        return array


@dataclass
class Batch(ImageArea):
    def __post_init__(self):
        super().__post_init__()
        self.downsample = self.slide.level_downsamples[self.level]

    @property
    def top_level_location(self):
        print(self.row, self.column, self.downsample)
        return (round(self.row * self.downsample),
                round(self.column * self.downsample))

    def get_patches(self,
                    patch_size: Tuple[int, int],
                    filter_: Filter = None) -> "Patch":
        dimensions = self.slide.level_dimensions[self.level][::-1]
        if filter_ is None:
            for row in range(0, self.size[0], patch_size[0]):
                if self.row + row + patch_size[0] > dimensions[0]:
                    continue
                for col in range(0, self.size[1], patch_size[1]):
                    if self.column + col + patch_size[1] > dimensions[1]:
                        continue
                    patch_row = int(row // patch_size[0])
                    patch_column = int(col // patch_size[1])
                    yield Patch(self.slide, patch_row, patch_column,
                                patch_size, self.image_info)

        else:
            index_patches = filter_.filter(self, patch_size)
            for p in np.argwhere(index_patches):
                patch = Patch(self.slide, p[0], p[1], patch_size,
                              self.image_info)
                logger.debug('yielding patch %s', patch)
                yield patch


@dataclass
class BatchIterator:
    slide: BasicSlide
    level: int
    n_batch: int
    image_info: ImageInfo
    batch_cls: Callable = Batch

    def __post_init__(self):
        self._level_dimensions = self.slide.level_dimensions[self.level][::-1]
        self._downsample = self.slide.level_downsamples[self.level]

        batch_size_0 = self._level_dimensions[0] // self.n_batch
        self._batch_size = (batch_size_0, self._level_dimensions[1])

    def __iter__(self):
        with Bar('collect batch') as bar:
            for i in range(0, self._level_dimensions[0], self._batch_size[0]):
                size_0 = min(self._batch_size[0],
                             self._level_dimensions[0] - i)
                #  if self._level_dimensions[0] - i < self._batch_size[0]:
                #      continue
                #  for j in range(0, self._level_dimensions[1], self._batch_size[1]):
                #      size = (size_0,
                #              min(self._batch_size[1],
                #                  self._level_dimensions[1] - j))
                bar.next()
                yield self.batch_cls(self.slide, i, 0, self.level,
                                     (size_0, self._level_dimensions[1]),
                                     self.image_info)

    def __len__(self):
        return self._level_dimensions[0] // self._batch_size[0]


@dataclass
class Patch(ImageArea):
    slide: BasicSlide
    row: int
    column: int
    level: int
    size: Tuple[int, int]
    PIL_format: bool = False

    def __post_init__(self):
        self._array = None

    @property
    def top_level_location(self):
        return (self.row * self.size[0], self.column * self.size[1])
