import abc
import logging
from datetime import datetime as dt

import dask.array as da
import numpy as np

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
                 round_to_0_100: bool = True,
                 max_MB_prediction=None) -> Mask:

        logger.info('classify: %s, %s, %s, %s, %s, %s', slide.filename,
                    filter_, threshold, level, max_MB_prediction,
                    round_to_0_100)
        #  batches = self._get_batch_iterator(slide, level, n_batch,
        #                                     self._color_type, self._coords,
        #                                     self._channel)
        if self._patch_size:
            predictions = self._classify_patches(slide, self._patch_size,
                                                 level, filter_, threshold,
                                                 round_to_0_100,
                                                 max_MB_prediction)
        else:
            predictions = self._classify_batches(slide, level, filter_,
                                                 max_MB_prediction)
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
                          round_to_0_100: bool = True,
                          max_MB_prediction=None) -> Mask:
        dimensions = slide.level_dimensions[level][::-1]
        dtype = 'uint8' if threshold or round_to_0_100 else 'float32'
        res = np.zeros(
            (dimensions[0] // patch_size[0], dimensions[1] // patch_size[1]),
            dtype=dtype)

        #  patch_indexes = filter_ if filter_ is not None else np.ndindex(
        #      dimensions[0] // patch_size[0], dimensions[1] // patch_size[1])
        #  patches_to_predict = [
        #      Patch(slide, p[0], p[1], level, patch_size, self._image_info)
        #      for p in patch_indexes
        #  ]
        #
        #
        #  predictions = []
        #  with Bar('Predictions', max=len(patches_to_predict) // n_patch
        #           or 1) as predict_bar:
        #      for i in range(0, len(patches_to_predict), n_patch):
        #          patches = patches_to_predict[i:i + n_patch]
        #          predictions.append(
        #              self._classify_array(
        #                  np.stack([p.array() for p in patches]), threshold,
        #                  round_to_0_100))
        #          predict_bar.next()
        #  if predictions:
        #      predictions = np.concatenate(predictions)
        #  for i, p in enumerate(predictions):
        #      patch = patches_to_predict[i]
        #      res[patch.row, patch.column] = p
        return res

    def _classify_batches(self, slide, level, filter_, max_MB_prediction):
        slide_array = self._get_slide_array(slide, level)
        if filter_ is None:
            return self._classify_batches_no_filter(slide_array,
                                                    max_MB_prediction)

        else:
            return self._classify_batches_with_filter(slide, slide_array,
                                                      level, filter_,
                                                      max_MB_prediction)

    def _get_slide_array(self, slide, level):
        return slide[level].convert(self.model.image_info)
        #  return slide[level]

    def _classify_batches_with_filter(self, slide, slide_array, level, filter_,
                                      max_MB_prediction):
        scale_factor = round(
            slide.level_downsamples[filter_.mask.extraction_level]) // round(
                slide.level_downsamples[level])

        res = np.zeros(slide_array.size, dtype='float32')
        max_contigous_pixels = round(max_MB_prediction * 10**6 // 3) \
            if max_MB_prediction is not None else slide_array.size[1]
        i = 0
        while i < len(filter_):
            contigous_pixels = 1
            for j in range(i + 1, len(filter_)):
                if filter_[i][0] == filter_[j][0] and filter_[j][1] - filter_[
                        i][1] == contigous_pixels:
                    contigous_pixels += 1
                    if contigous_pixels >= max_contigous_pixels:
                        break
                else:
                    break

            pixel = filter_[i] * scale_factor
            area = slide_array[pixel[0]:pixel[0] + scale_factor,
                               pixel[1]:pixel[1] +
                               scale_factor * contigous_pixels]

            n_px = area.size[0] * area.size[1]
            prediction = self._predict(area.reshape((n_px, )))
            res[pixel[0]:pixel[0] + scale_factor, pixel[1]:pixel[1] +
                scale_factor * contigous_pixels] = prediction.reshape(
                    (scale_factor, scale_factor * contigous_pixels))
            i = i + contigous_pixels

        return res

    def _classify_batches_no_filter(self, slide_array, max_MB_prediction):
        predictions = []
        step = slide_array.size[0] if max_MB_prediction is None else round(
            max_MB_prediction * 10**6 // (3 * slide_array.size[1]))
        logger.info('max_MB_prediction %s, size_1 %s step %s, n_batch %s',
                    max_MB_prediction, slide_array.size[1], step,
                    slide_array.size[1] // step)
        for i in range(0, slide_array.size[0], step):
            area = slide_array[i:i + step, :]
            n_px = area.size[0] * area.size[1]
            area_reshaped = area.reshape((n_px, ))

            prediction = self._predict(area_reshaped)

            prediction = prediction.reshape(area.size[0], area.size[1])
            predictions.append(prediction)
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
    def _concatenate(seq, axis):
        return np.concatenate(seq, axis)
