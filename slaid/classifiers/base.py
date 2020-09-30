import abc
from typing import Tuple

import numpy as np

from slaid.commons import Image, Mask, Slide
from slaid.models import Model


class Classifier(abc.ABC):
    @abc.abstractmethod
    def classify_patch(
        self,
        slide,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        threshold: float = 0.8,
    ) -> np.ndarray:
        pass

    @abc.abstractmethod
    def classify(self,
                 slide: Slide,
                 patch_filter=None,
                 threshold: float = 0.8,
                 level: int = 2,
                 patch_size=None):
        pass


class BasicClassifier(Classifier):
    def __init__(self, model: "Model", feature: str):
        self._model = model
        self._feature = feature

    def classify(self,
                 slide: Slide,
                 patch_filter=None,
                 threshold: float = 0.8,
                 level: int = 2,
                 patch_size=None):

        if patch_filter is not None:
            assert patch_size is not None

        if patch_size is not None:
            dimensions = slide.level_dimensions[level]
            mask = np.zeros(dimensions[::-1], dtype='uint8')
            for p in slide.patches(level, patch_size):
                patch_mask = self.classify_patch(slide, (p.x, p.y), level,
                                                 p.size)
                shape = patch_mask.shape[::-1]
                mask[p.y:p.y + shape[1], p.x:p.x + shape[0]] = patch_mask
        #  if patch_filter:
        #      assert patch_size is not None
        #      patches = slide.patches.filter(patch_filter)
        #      for patch in patches:
        #          self._classify_patch(patch, slide, patch_area, threshold,
        #                               patch_threshold)
        #  else:
        #      self._classify_whole_slide(slide, patch_area, threshold,
        #                                 patch_threshold, include_mask)
        else:
            mask = self.classify_patch(slide, (0, 0), level,
                                       slide.level_dimensions[level],
                                       threshold)

        slide.masks[self._feature] = Mask(mask, level,
                                          slide.level_downsamples[level])

    def classify_patch(
        self,
        slide,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        threshold: float = 0.8,
    ) -> np.ndarray:
        image = slide.read_region(location, level, size)
        image_array = self._get_image_array(image)
        prediction = self._model.predict(image_array)
        return self._get_mask(prediction, size[::-1], threshold)

    def _get_image_array(self, image: Image) -> np.ndarray:
        image_array = image.to_array(True)
        n_px = image_array.shape[0] * image_array.shape[1]
        image_array = image_array[:, :, :3].reshape(n_px, 3)
        return image_array

    def _get_mask(self, prediction: np.ndarray, shape: Tuple[int, int],
                  threshold: float) -> np.ndarray:
        mask = prediction.reshape(shape)
        mask[mask < threshold] = 0
        mask[mask > threshold] = 1
        mask = np.array(mask, dtype=np.uint8)
        return mask
