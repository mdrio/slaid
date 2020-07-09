import abc
from commons import Patch
from typing import Dict, List, Tuple, Callable
from commons import Slide
import numpy as np
from tifffile import imwrite
from PIL import Image, ImageDraw, ImageFont


class PatchFeature:
    def __init__(self, patch: Patch, data: Dict):
        self.patch = patch
        self.data = data

    def __str__(self):
        return f'{self.patch}, data: {self.data}'


class FeatureTIFFRenderer(abc.ABC):
    @abc.abstractmethod
    def render(self, filename: str, features: List[PatchFeature]):
        pass


def karolinska_rgb_convert(features: List[PatchFeature]) -> np.array:
    for feature in features:
        cancer_percentage = feature.data[
            KarolinskaDummyClassifier.Feature.CANCER_PERCENTAGE]
        mask_value = int(round(cancer_percentage * 255))
        data = (mask_value, mask_value, mask_value,
                255) if cancer_percentage > 0 else (0, 0, 0, 0)
        yield np.full(feature.patch.size + (4, ), data, 'uint8')


def karolinska_text_convert(features: List[PatchFeature]) -> np.array:
    for feature in features:
        cancer_percentage = feature.data[
            KarolinskaDummyClassifier.Feature.CANCER_PERCENTAGE]
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
    @abc.abstractmethod
    def classify_patch(patch: Patch) -> PatchFeature:
        pass


class KarolinskaDummyClassifier(Classifier):
    class Feature:
        CANCER_PERCENTAGE = 'cancer_percentage'

    def __init__(self, mask: Slide):
        self.mask = mask

    def classify_patch(self, patch: Patch) -> PatchFeature:
        image = self.mask.read_region(location=(patch.x, patch.y),
                                      level=0,
                                      size=patch.size)

        data = np.array(image.getchannel(0).getdata())
        features = PatchFeature(
            patch, {
                KarolinskaDummyClassifier.Feature.CANCER_PERCENTAGE:
                sum(map(lambda el: 1 if el == 2 else 0, data)) / len(data)
            })
        return features

    def classify(self,
                 patch_size: Tuple[int, int] = None) -> List[PatchFeature]:
        features = []
        for patch in self.mask.iterate_by_patch(patch_size):
            features.append(self.classify_patch(patch))
        return features