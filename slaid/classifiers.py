import abc
import pickle
from typing import Tuple, Dict
from slaid.commons import Patch, Slide, get_class, round_to_patch,\
    PatchCollection, PATCH_SIZE
import numpy as np
from PIL import Image
import random
import os
from multiprocessing import Pool


class Classifier(abc.ABC):
    @abc.abstractstaticmethod
    def create(*args):
        pass

    @abc.abstractmethod
    def classify_patch(self, patch: Patch, *args, **kwargs) -> Dict:
        pass

    def classify(self,
                 slide: Slide,
                 patch_filter=None,
                 *args,
                 **kwargs) -> Slide:
        patches = slide.patches if patch_filter is None else\
            slide.patches.filter(patch_filter)

        for patch in patches:
            features = self.classify_patch(patch, *args, **kwargs)
            slide.patches.update_patch(patch=patch, features=features)
        return slide


class KarolinskaFeature:
    CANCER_PERCENTAGE = 'cancer_percentage'


class KarolinskaRandomClassifier(Classifier):
    @staticmethod
    def create(*args):
        return KarolinskaRandomClassifier()

    def classify_patch(self, patch: Patch) -> Dict:
        return {KarolinskaFeature.CANCER_PERCENTAGE: random.random()}


class KarolinskaTrueValueClassifier(Classifier):
    def __init__(self, mask: Slide):
        self.mask = mask

    @staticmethod
    def create(mask_filename):
        return KarolinskaTrueValueClassifier(Slide(mask_filename))

    def classify_patch(self, patch: Patch, extraction_level: int = 2) -> Dict:
        image = self.mask.read_region(location=(patch.x, patch.y),
                                      size=patch.size)

        data = np.array(image.getchannel(0).getdata())
        return {
            KarolinskaFeature.CANCER_PERCENTAGE:
            sum(map(lambda el: 1 if el == 2 else 0, data)) / len(data)
        }


class ParallelClassifier(Classifier):
    def __init__(self, classifier: Classifier):
        self._classifier = classifier

    @staticmethod
    def create(classifier_cls_name: str, *args):
        return ParallelClassifier(
            get_class(classifier_cls_name, 'classifiers').create(*args))

    def classify_patch(self, patch: Patch) -> Dict:
        return self._classifier.classify_patch(patch)

    def _classify_patch(self, patch) -> Tuple[Patch, Dict]:
        return (patch, self.classify_patch(patch))

    def classify(self, patch_collection: PatchCollection) -> PatchCollection:
        with Pool(os.cpu_count()) as pool:
            patch_features = pool.map(self._classify_patch,
                                      iter(patch_collection))

        for patch, features in patch_features:
            patch_collection.update_patch(patch=patch, features=features)

        return patch_collection


class TissueMaskPredictor(abc.ABC):
    @abc.abstractmethod
    def get_tissue_mask(image: Image, threshold: float) -> np.array:
        pass


class Model(abc.ABC):
    @abc.abstractmethod
    def predict(self, array: np.array) -> np.array:
        pass


class BasicTissueMaskPredictor(TissueMaskPredictor):
    @staticmethod
    def create(model_filename):
        with open(model_filename, 'rb') as f:
            return BasicTissueMaskPredictor(pickle.load(f))

    def __init__(self, model):
        self._model = model

    def get_tissue_mask(self, image: Image, threshold: float) -> np.array:
        np_img = np.array(image)
        n_px = np_img.shape[0] * np_img.shape[1]
        x = np_img[:, :, :3].reshape(n_px, 3)

        #  if self.ann_model:
        #      pred = self.model.predict(x, batch_size=self.bs)
        #  else:
        #      pred = self.model.predict(x)
        #
        pred = self._model.predict(x)
        msk_pred = pred.reshape(np_img.shape[0], np_img.shape[1])
        msk_pred[msk_pred < threshold] = 0
        msk_pred[msk_pred > threshold] = 1

        return msk_pred


class TissueFeature:
    TISSUE_PERCENTAGE = 'tissue_percentage'
    TISSUE_MASK = 'tissue_mask'


class TissueClassifier(Classifier):
    def __init__(self, predictor: TissueMaskPredictor):
        self._predictor = predictor

    @staticmethod
    def create(model_filename,
               predictor_cls_name: str = 'BasicTissueMaskPredictor'):

        return TissueClassifier(
            get_class(predictor_cls_name,
                      'slaid.classifiers').create(model_filename))
        raise NotImplementedError()

    def _get_mask_tissue_from_slide(self, slide, threshold):

        dims = slide.level_dimensions
        ds = [int(i) for i in slide.level_downsamples]

        x0 = 0
        y0 = 0
        x1 = dims[0][0]
        y1 = dims[0][1]

        delta_x = x1 - x0
        delta_y = y1 - y0

        pil_img = slide.read_region(
            location=(x0, y0),
            size=(delta_x // ds[slide.extraction_level],
                  delta_y // ds[slide.extraction_level]))
        return self._predictor.get_tissue_mask(pil_img, threshold)

    def _get_tissue_patches_coordinates(
        self,
        slide,
        tissue_mask,
        patch_size=PATCH_SIZE,
    ):

        tissue_mask *= 255
        # resize and use to extract patches
        lev = slide.get_best_level_for_downsample(16)
        lev_dim = slide.level_dimensions[lev]

        big_x = slide.level_dimensions[slide.patches.extraction_level][0]
        big_y = slide.level_dimensions[slide.patches.extraction_level][1]

        # downsampling factor of level0 with respect  patch size
        dim_x, dim_y = patch_size
        xx = round(big_x / dim_x)
        yy = round(big_y / dim_y)

        mask = Image.new('L', lev_dim)
        mask.putdata(tissue_mask.flatten())
        mask = mask.resize((xx, yy), resample=Image.BILINEAR)
        tissue = [(x, y) for x in range(xx) for y in range(yy)
                  if mask.getpixel((x, y)) > 0]

        ext_lev_ds = slide.level_downsamples[slide.patches.extraction_level]
        return [
            round_to_patch((round(x * big_x / xx * ext_lev_ds),
                            round(y * big_y / yy * ext_lev_ds)), patch_size)
            for (x, y) in tissue
        ]

    def classify_patch(self, patch: Patch) -> Patch:
        raise NotImplementedError

    def classify(self,
                 slide: Slide,
                 pixel_threshold: float = 0.8,
                 minimum_tissue_ratio: float = 0.01,
                 downsampling: int = 16,
                 include_mask_feature=False) -> Slide:

        lev = slide.get_best_level_for_downsample(downsampling)
        lev_dim = slide.level_dimensions[lev]
        thumb = slide.read_region(location=(0, 0), size=lev_dim)
        tissue_mask = self._predictor.get_tissue_mask(thumb, pixel_threshold)

        dim_x, dim_y = slide.patches.patch_size
        patch_area = dim_x * dim_y
        #  patch_area_th = patch_area * self._patch_threshold

        patch_coordinates = self._get_tissue_patches_coordinates(
            slide,
            tissue_mask,
            slide.patches.patch_size,
        )

        slide.patches.add_feature(TissueFeature.TISSUE_PERCENTAGE, 0.0)
        if include_mask_feature:
            slide.patches.add_feature(TissueFeature.TISSUE_MASK)

        for (coor_x, coor_y) in patch_coordinates:
            patch = slide.read_region(location=(coor_x, coor_y),
                                      size=(dim_x, dim_y))

            tissue_mask = self._predictor.get_tissue_mask(
                patch, pixel_threshold)

            tissue_area = np.sum(tissue_mask)
            tissue_ratio = tissue_area / patch_area
            if tissue_ratio > minimum_tissue_ratio:
                features = {TissueFeature.TISSUE_PERCENTAGE: tissue_ratio}
                if include_mask_feature:
                    features[TissueFeature.TISSUE_MASK] = np.array(tissue_mask,
                                                                   dtype=bool)
                slide.patches.update_patch((coor_x, coor_y), features=features)

        return slide
