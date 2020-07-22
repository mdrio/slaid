import abc
from commons import Patch, Slide, get_class, round_to_patch
from typing import List, Tuple, Callable, Any, Dict
import numpy as np
from tifffile import imwrite
from PIL import Image
import random
import os
from multiprocessing import Pool
import json
import pandas as pd
from collections import defaultdict, OrderedDict


class PatchCollection(abc.ABC):
    @abc.abstractclassmethod
    def from_pandas(cls, slide: Slide, patch_size: Tuple[int, int],
                    dataframe: pd.DataFrame):
        pass

    def __init__(self, slide: Slide, patch_size: Tuple[int, int]):
        self._slide = slide
        self._patch_size = patch_size

    @property
    def slide(self) -> Slide:
        return self._slide

    @property
    def patch_size(self) -> Tuple[int, int]:
        return self._patch_size

    @abc.abstractproperty
    def dataframe(self) -> pd.DataFrame:
        pass

    @abc.abstractproperty
    def loc(self) -> "PatchCollection":
        pass

    @abc.abstractmethod
    def get_patch(self, coordinates: Tuple[int, int]) -> Patch:
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def update_patch(self,
                     coordinates: Tuple[int, int] = None,
                     patch: Patch = None,
                     features: Dict = None):
        pass

    @abc.abstractproperty
    def features(self) -> List[str]:
        pass

    @abc.abstractmethod
    def add_feature(self, feature: str, default_value: Any = None):
        pass

    @abc.abstractmethod
    def update(self, other_collection: 'PatchCollection'):
        pass

    @abc.abstractmethod
    def merge(self, other_collection: 'PatchCollection'):
        pass


class PandasPatchCollection(PatchCollection):
    class LocIndexer:
        def __init__(self, collection: "PandasPatchCollection"):
            self.collection = collection

        def __getitem__(self, key):
            return PandasPatchCollection.from_pandas(
                self.collection.slide, self.collection.patch_size,
                self.collection.dataframe.loc[key])

    @classmethod
    def from_pandas(cls, slide: Slide, patch_size: Tuple[int, int],
                    dataframe: pd.DataFrame):
        patch_collection = cls(slide, patch_size)
        # TODO add some assert to verify dataframe is compatible
        patch_collection._dataframe = dataframe
        return patch_collection

    def __init__(self, slide: Slide, patch_size: Tuple[int, int]):
        super().__init__(slide, patch_size)
        self._dataframe = self._init_dataframe()
        self._loc = PandasPatchCollection.LocIndexer(self)

    def _init_dataframe(self):
        data = defaultdict(lambda: [])
        for p in self._slide.iterate_by_patch(self._patch_size):
            data['y'].append(p.y)
            data['x'].append(p.x)
        df = pd.DataFrame(data, dtype=int)
        df.set_index(['y', 'x'], inplace=True)
        return df

    def __getitem__(self, key):
        return self._dataframe[key]

    def _create_patch(self, coordinates: Tuple[int, int],
                      data: Tuple) -> Patch:
        y, x = coordinates
        features = dict(data)
        return Patch(self.slide, (x, y), self.patch_size, features)

    def __iter__(self):
        for data in self._dataframe.iterrows():
            yield self._create_patch(data[0], data[1])

    def __len__(self):
        return len(self._dataframe)

    @property
    def dataframe(self):
        return self._dataframe

    def get_patch(self, coordinates: Tuple[int, int]) -> Patch:
        return self._create_patch(coordinates[::-1],
                                  self._dataframe.loc[coordinates[::-1]])

    @property
    def features(self) -> List[str]:
        return [c for c in self._dataframe.columns[2:]]

    def add_feature(self, feature: str, default_value: Any = None):
        if feature not in self.features:
            self.dataframe.insert(len(self._dataframe.columns), feature,
                                  default_value)

    def update_patch(self,
                     coordinates: Tuple[int, int] = None,
                     patch: Patch = None,
                     features: Dict = None):
        if patch:
            coordinates = (patch.x, patch.y)

        features = OrderedDict(features)
        if coordinates is None:
            raise RuntimeError('coordinates and patch cannot be None')

        missing_features = features.keys() - set(self._dataframe.columns)
        for f in missing_features:
            self.add_feature(f)
        self._dataframe.loc[coordinates[::-1],
                            list(features.keys())] = list(features.values())

    @property
    def loc(self) -> "PatchCollection":
        return self._loc

    def update(self, other: "PandasPatchCollection"):
        self.dataframe.update(other.dataframe)

    def merge(self, other_collection: 'PandasPatchCollection'):
        self._dataframe = self.dataframe.merge(other_collection.dataframe,
                                               'left').set_index(
                                                   self._dataframe.index)


class JSONEncoder(json.JSONEncoder):
    def _encode_patch(self, patch: Patch):

        return {
            'slide': patch.slide.ID,
            'x': patch.x,
            'y': patch.y,
            'size': patch.size,
            'features': patch.features
        }

    def default(self, obj):
        if isinstance(obj, Patch):
            return self._encode_patch(obj)
        elif isinstance(obj, PatchCollection):
            return [self._encode_patch(p) for p in obj]
        return super().default(obj)


class FeatureTIFFRenderer(abc.ABC):
    @abc.abstractmethod
    def render(self, filename: str, patches: List[Patch]):
        pass


def karolinska_rgb_convert(patches: PatchCollection) -> np.array:
    for patch in patches:
        cancer_percentage = patch.features[KarolinskaFeature.CANCER_PERCENTAGE]
        cancer_percentage = 0 if np.isnan(
            cancer_percentage) else cancer_percentage
        mask_value = int(round(cancer_percentage * 255))
        data = (mask_value, 0, 0, 255) if cancer_percentage > 0 else (0, 0, 0,
                                                                      0)
        yield np.full(patch.size + (4, ), data, 'uint8')


class BasicFeatureTIFFRenderer:
    def __init__(
        self,
        rgb_convert: Callable,
        shape: Tuple[int, int],
    ):
        self._shape = shape
        self._rgb_convert = rgb_convert

    def render(self, filename: str, patches: PatchCollection):
        imwrite(filename,
                self._rgb_convert(patches),
                dtype='uint8',
                shape=(self._shape[1], self._shape[0], 4),
                photometric='rgb',
                tile=patches.patch_size,
                extrasamples=('ASSOCALPHA', ))


class Classifier(abc.ABC):
    @abc.abstractstaticmethod
    def create(*args):
        pass

    @abc.abstractmethod
    def classify_patch(self, patch: Patch) -> Dict:
        pass

    def classify(self, patch_collection: PatchCollection) -> PatchCollection:
        for patch in patch_collection:
            features = self.classify_patch(patch)
            patch_collection.update_patch(patch=patch, features=features)
        return patch_collection


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

    def classify_patch(self, patch: Patch) -> Dict:
        image = self.mask.read_region(location=(patch.x, patch.y),
                                      level=0,
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


class BasicTissueMaskPredictor(TissueMaskPredictor):
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


class TissueClassifier(Classifier):
    def __init__(self,
                 predictor: TissueMaskPredictor,
                 mask_threshold: float = 0.5,
                 patch_threshold: float = 0.8,
                 level: int = 2):
        self._predictor = predictor
        self._mask_threshold = mask_threshold
        self._patch_threshold = patch_threshold
        self._level = level

    @staticmethod
    def create(predictor_cls_name: str, *args):
        #  return TissueClassifier(
        #      get_class(predictor_cls_name, 'classifiers').create(*args))
        raise NotImplementedError()

    def _get_mask_tissue_from_slide(self, slide, threshold, level=2):

        dims = slide.level_dimensions
        ds = [int(i) for i in slide.level_downsamples]

        x0 = 0
        y0 = 0
        x1 = dims[0][0]
        y1 = dims[0][1]

        delta_x = x1 - x0
        delta_y = y1 - y0

        pil_img = slide.read_region(location=(x0, y0),
                                    level=level,
                                    size=(delta_x // ds[level],
                                          delta_y // ds[level]))
        return self._predictor.get_tissue_mask(pil_img, threshold)

    def _get_tissue_patches_coordinates(self,
                                        slide,
                                        tissue_mask,
                                        patch_size,
                                        extraction_lev=0):

        tissue_mask *= 255
        # resize and use to extract patches
        lev = slide.get_best_level_for_downsample(16)
        lev_dim = slide.level_dimensions[lev]

        big_x = slide.level_dimensions[extraction_lev][0]
        big_y = slide.level_dimensions[extraction_lev][1]

        # downsampling factor of level0 with respect  patch size
        dim_x, dim_y = patch_size
        xx = round(big_x / dim_x)
        yy = round(big_y / dim_y)

        mask = Image.new('L', lev_dim)
        mask.putdata(tissue_mask.flatten())
        mask = mask.resize((xx, yy), resample=Image.BILINEAR)
        tissue = [(x, y) for x in range(xx) for y in range(yy)
                  if mask.getpixel((x, y)) > 0]
        random.shuffle(tissue)  # randomly permute the elements

        ext_lev_ds = slide.level_downsamples[extraction_lev]
        return [
            round_to_patch((round(x * big_x / xx * ext_lev_ds),
                            round(y * big_y / yy * ext_lev_ds)), patch_size)
            for (x, y) in tissue
        ]

    def classify_patch(self, patch: Patch) -> Patch:
        raise NotImplementedError

    def classify(self, patch_collection: PatchCollection) -> PatchCollection:

        lev = patch_collection.slide.get_best_level_for_downsample(16)
        lev_dim = patch_collection.slide.level_dimensions[lev]
        thumb = patch_collection.slide.read_region(location=(0, 0),
                                                   level=lev,
                                                   size=lev_dim)
        tissue_mask = self._predictor.get_tissue_mask(thumb,
                                                      self._mask_threshold)

        dim_x, dim_y = patch_collection.patch_size
        patch_area = dim_x * dim_y
        #  patch_area_th = patch_area * self._patch_threshold
        extraction_lev = 0  # TODO check it is correct

        patch_coordinates = self._get_tissue_patches_coordinates(
            patch_collection.slide, tissue_mask, patch_collection.patch_size)

        patch_collection.add_feature(TissueFeature.TISSUE_PERCENTAGE, 0.0)

        for (coor_x, coor_y) in patch_coordinates:
            patch = patch_collection.slide.read_region(location=(coor_x,
                                                                 coor_y),
                                                       level=extraction_lev,
                                                       size=(dim_x, dim_y))

            tissue_area = np.sum(
                self._predictor.get_tissue_mask(patch, self._patch_threshold))

            patch_collection.update_patch((coor_x, coor_y),
                                          features={
                                              TissueFeature.TISSUE_PERCENTAGE:
                                              tissue_area / patch_area
                                          })

        return patch_collection
