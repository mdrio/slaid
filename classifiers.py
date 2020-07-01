from abc import ABC
from commons import Patch
from typing import Dict
from commons import Slide
import numpy as np


class FeatureSet(object):
    def __init__(self, patch: Patch, features=Dict):
        self.patch = patch
        self.features = features

    def __str__(self):
        return f'{self.patch}, features: {self.features}'


class Classifier(ABC):
    def classify_patch(patch: Patch) -> FeatureSet:
        pass


class KarolinskaDummyClassifier(Classifier):
    def __init__(self, mask: Slide):
        self.mask = mask

    def classify_patch(self, patch: Patch) -> FeatureSet:
        image = self.mask.read_region(location=(patch.x, patch.y),
                                      level=0,
                                      size=(patch.size.x, patch.size.y))

        data = np.array(image.getchannel(0).getdata())
        feature_set = FeatureSet(
            patch, {
                "cancer_percentage":
                sum(map(lambda el: 1 if el == 2 else 0, data)) / len(data)
            })
        return feature_set
