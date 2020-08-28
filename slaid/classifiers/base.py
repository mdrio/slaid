import abc
import pickle
from typing import Dict, Tuple

import numpy as np

from slaid.commons import Patch, Slide


class Classifier(abc.ABC):
    @abc.abstractclassmethod
    def create(cls, *args):
        pass

    @abc.abstractmethod
    def classify_patch(self, patch: Patch, *args, **kwargs) -> Dict:
        pass

    @abc.abstractmethod
    def classify(
        self,
        slide: Slide,
        mask_threshold: float = 0.8,
        patch_threshold: float = 0.8,
        include_mask: bool = False,
    ):
        pass


class BasicClassifier:
    def __init__(self, model: "Model", feature: str):
        self._model = model
        self._feature = feature

    def classify(
        self,
        slide: Slide,
        mask_threshold: float = 0.8,
        patch_threshold: float = 0.8,
        include_mask: bool = False,
    ):

        image_array = self._get_image_array(slide)
        prediction = self._model.predict(image_array)

        mask = self._get_mask(prediction, slide.dimensions_at_extraction_level,
                              mask_threshold)

        slide.patches.add_feature(self._feature, 0.0)
        if include_mask:
            slide.masks[self._feature] = mask

        patch_area = slide.patches.patch_size[0] * slide.patches.patch_size[1]
        for patch in slide.patches:
            self._update_patch(slide, patch, mask, patch_area, patch_threshold,
                               include_mask)

    def _get_image_array(self, slide: Slide) -> np.ndarray:
        image = slide.read_region((0, 0), slide.dimensions_at_extraction_level)
        image_array = image.to_array(True)
        n_px = image_array.shape[0] * image_array.shape[1]
        image_array = image_array[:, :, :3].reshape(n_px, 3)
        return image_array

    def _get_mask(self, prediction: np.ndarray, shape: Tuple[int, int],
                  threshold: float) -> np.ndarray:
        mask = prediction.reshape(*shape)
        mask[mask < threshold] = 0
        mask[mask > threshold] = 1
        mask = mask.transpose()
        return mask

    def _update_patch(self, slide, patch: Patch, mask: np.ndarray,
                      patch_area: float, threshold: float,
                      include_mask_feature):
        patch_mask = mask[patch.x:patch.x + patch.size[0],
                          patch.y:patch.y + patch.size[1]]
        tissue_area = np.sum(patch_mask)
        tissue_ratio = tissue_area / patch_area
        if tissue_ratio > threshold:
            features = {self._feature: tissue_ratio}

            if include_mask_feature:
                # FIXME, duplicate code
                feature_mask = f'{self._feature}_mask'
                features[feature_mask] = np.array(patch_mask, dtype=np.uint8)
            slide.patches.update_patch(patch=patch, features=features)


class Model:
    def __init__(self, filename: str):
        with open(filename, 'rb') as f:
            self._model = pickle.load(f)

    def predict(self, array: np.array) -> np.array:
        return self._model.predict(array)


#  class InterpolatedTissueClassifier(TissueClassifier):
#      def _get_mask_tissue_from_slide(self, slide, threshold):
#
#          dims = slide.level_dimensions
#          ds = [int(i) for i in slide.level_downsamples]
#
#          x0 = 0
#          y0 = 0
#          x1 = dims[0][0]
#          y1 = dims[0][1]
#
#          delta_x = x1 - x0
#          delta_y = y1 - y0
#
#          pil_img = slide.read_region(
#              location=(x0, y0),
#              size=(delta_x // ds[slide.extraction_level],
#                    delta_y // ds[slide.extraction_level]))
#          return self._predictor.get_tissue_mask(pil_img, threshold)
#
#      def _get_tissue_patches_coordinates(
#          self,
#          slide,
#          tissue_mask,
#          patch_size=PATCH_SIZE,
#      ):
#
#          tissue_mask *= 255
#          # resize and use to extract patches
#          lev = slide.get_best_level_for_downsample(16)
#          lev_dim = slide.level_dimensions[lev]
#
#          big_x = slide.level_dimensions[slide.patches.extraction_level][0]
#          big_y = slide.level_dimensions[slide.patches.extraction_level][1]
#
#          # downsampling factor of level0 with respect  patch size
#          dim_x, dim_y = patch_size
#          xx = round(big_x / dim_x)
#          yy = round(big_y / dim_y)
#
#          mask = PIL_Image.new('L', lev_dim)
#          mask.putdata(tissue_mask.flatten())
#          mask = mask.resize((xx, yy), resample=PIL_Image.BILINEAR)
#          tissue = [(x, y) for x in range(xx) for y in range(yy)
#                    if mask.getpixel((x, y)) > 0]
#
#          ext_lev_ds = slide.level_downsamples[slide.patches.extraction_level]
#          return [
#              round_to_patch((round(x * big_x / xx * ext_lev_ds),
#                              round(y * big_y / yy * ext_lev_ds)), patch_size)
#              for (x, y) in tissue
#          ]
#
#      def classify_patch(self, patch: Patch) -> Patch:
#          raise NotImplementedError
#
#      def classify(self,
#                   slide: Slide,
#                   pixel_threshold: float = 0.8,
#                   minimum_tissue_ratio: float = 0.01,
#                   downsampling: int = 16,
#                   include_mask_feature=False) -> Slide:
#
#          lev = slide.get_best_level_for_downsample(downsampling)
#          lev_dim = slide.level_dimensions[lev]
#          thumb = slide.read_region(location=(0, 0), size=lev_dim)
#          tissue_mask = self._predictor.get_tissue_mask(thumb, pixel_threshold)
#
#          dim_x, dim_y = slide.patches.patch_size
#          patch_area = dim_x * dim_y
#          #  patch_area_th = patch_area * self._patch_threshold
#
#          patch_coordinates = self._get_tissue_patches_coordinates(
#              slide,
#              tissue_mask,
#              slide.patches.patch_size,
#          )
#
#          slide.patches.add_feature(TissueFeature.TISSUE_PERCENTAGE, 0.0)
#          if include_mask_feature:
#              slide.patches.add_feature(TissueFeature.TISSUE_MASK)
#
#          for (coor_x, coor_y) in patch_coordinates:
#              patch = slide.read_region(location=(coor_x, coor_y),
#                                        size=(dim_x, dim_y))
#
#              tissue_mask = self._predictor.get_tissue_mask(
#                  patch, pixel_threshold)
#
#              tissue_area = np.sum(tissue_mask)
#              tissue_ratio = tissue_area / patch_area
#              if tissue_ratio > minimum_tissue_ratio:
#                  features = {TissueFeature.TISSUE_PERCENTAGE: tissue_ratio}
#                  if include_mask_feature:
#                      features[TissueFeature.TISSUE_MASK] = np.array(tissue_mask,
#                                                                     dtype=bool)
#                  slide.patches.update_patch((coor_x, coor_y), features=features)
#
#          return slide
