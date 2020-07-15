import abc
from commons import Patch, SlideIterator, get_class
from typing import Dict, List, Tuple, Callable
from commons import Slide
import numpy as np
from tifffile import imwrite
from PIL import Image, ImageDraw, ImageFont
import random
import os
from multiprocessing import Pool
import json


class PatchFeature:
    def __init__(self, patch: Patch, data: Dict):
        self.patch = patch
        self.data = data

    @property
    def size(self):
        return self.patch.size

    @property
    def x(self):
        return self.patch.x

    @property
    def y(self):
        return self.patch.y

    def __eq__(self, other):
        return self.x == other.x and self.y == self.y and\
            self.size == other.size and self.data == other.data

    def __repr__(self):
        return f'{self.x}, {self.y}, {self.size}, {self.data}'


class PatchFeatureCollection:
    def __init__(self, slide: Slide, patch_size, features: List[PatchFeature]):
        self.slide = slide
        self.patch_size = patch_size
        self.features = features

    def __getitem__(self, key):
        return self.features.__getitem__(key)

    def __iter__(self):
        return self.features

    def sort(self):
        self.features.sort(key=lambda e: e.patch.index)


class JSONEncoder(json.JSONEncoder):
    def _encode_patch_feature(self, feature: PatchFeature):

        return {
            'slide': feature.patch.slide.ID,
            'x': feature.patch.x,
            'y': feature.patch.y,
            'size': feature.patch.size,
            'data': feature.data
        }

    def default(self, obj):
        if isinstance(obj, PatchFeature):
            return self._encode_patch_feature(obj)
        elif isinstance(obj, PatchFeatureCollection):
            return [self._encode_patch_feature(f) for f in obj.features]
        return super().default(obj)


class FeatureTIFFRenderer(abc.ABC):
    @abc.abstractmethod
    def render(self, filename: str, features: List[PatchFeature]):
        pass


def karolinska_rgb_convert(features: List[PatchFeature]) -> np.array:
    for feature in features:
        cancer_percentage = feature.data[KarolinskaFeature.CANCER_PERCENTAGE]
        mask_value = int(round(cancer_percentage * 255))
        data = (mask_value, 0, 0, 255) if cancer_percentage > 0 else (0, 0, 0,
                                                                      0)
        yield np.full(feature.patch.size + (4, ), data, 'uint8')


def karolinska_text_convert(features: List[PatchFeature]) -> np.array:
    for feature in features:
        cancer_percentage = feature.data[KarolinskaFeature.CANCER_PERCENTAGE]
        red = int(round(cancer_percentage * 255))
        txt = Image.new("RGBA", feature.patch.size, (0, 0, 0, 0))
        fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 30)
        d = ImageDraw.Draw(txt)
        d.text((10, 10), f'{red}', font=fnt, fill=(255, 0, 0, 255))
        yield np.asarray(txt)


class BasicFeatureTIFFRenderer:
    def __init__(
        self,
        rgb_convert: Callable,
        shape: Tuple[int, int],
    ):
        self._shape = shape
        self._rgb_convert = rgb_convert

    def render(self, filename: str, features: List[PatchFeature]):
        imwrite(filename,
                self._rgb_convert(features),
                dtype='uint8',
                shape=(self._shape[1], self._shape[0], 4),
                photometric='rgb',
                tile=features[0].patch.size,
                extrasamples=('ASSOCALPHA', ))


class Classifier(abc.ABC):
    @abc.abstractstaticmethod
    def create(*args):
        pass

    @abc.abstractmethod
    def classify_patch(self, patch: Patch) -> PatchFeature:
        pass

    def classify(self,
                 slide: Slide,
                 patch_size: Tuple[int, int] = None) -> PatchFeatureCollection:
        features = []
        for patch in slide.iterate_by_patch(patch_size):
            features.append(self.classify_patch(patch))
        return PatchFeatureCollection(slide, patch_size, features)


class KarolinskaFeature:
    CANCER_PERCENTAGE = 'cancer_percentage'


class KarolinskaRandomClassifier(Classifier):
    @staticmethod
    def create(*args):
        return KarolinskaRandomClassifier()

    def classify_patch(self, patch: Patch) -> PatchFeature:
        feature = PatchFeature(
            patch, {KarolinskaFeature.CANCER_PERCENTAGE: random.random()})
        return feature


class KarolinskaTrueValueClassifier(Classifier):
    def __init__(self, mask: Slide):
        self.mask = mask

    @staticmethod
    def create(mask_filename):
        return KarolinskaTrueValueClassifier(Slide(mask_filename))

    def classify_patch(self, patch: Patch) -> PatchFeature:
        image = self.mask.read_region(location=(patch.x, patch.y),
                                      level=0,
                                      size=patch.size)

        data = np.array(image.getchannel(0).getdata())
        feature = PatchFeature(
            patch, {
                KarolinskaFeature.CANCER_PERCENTAGE:
                sum(map(lambda el: 1 if el == 2 else 0, data)) / len(data)
            })
        return feature


class ParallelClassifier(Classifier):
    def __init__(self, classifier: Classifier):
        self._classifier = classifier

    @staticmethod
    def create(classifier_cls_name: str, *args):
        return ParallelClassifier(
            get_class(classifier_cls_name, 'classifiers').create(*args))

    def classify_patch(self, patch: Patch) -> PatchFeature:
        return self._classifier(patch)

    def classify(self,
                 slide: Slide,
                 patch_size: Tuple[int, int] = None) -> PatchFeatureCollection:
        with Pool(os.cpu_count()) as pool:
            return PatchFeatureCollection(
                slide, patch_size,
                pool.map(self._classifier.classify_patch,
                         SlideIterator(slide, patch_size)))
