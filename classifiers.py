from abc import ABC
from patches import Patch
from typing import Dict
from openslide import OpenSlide
import numpy as np


class FeatureSet(object):
    def __init__(self, patch: Patch, features=Dict):
        self.patch = patch
        self.features = features


class Classifier(ABC):
    def classify_patch(patch: Patch) -> FeatureSet:
        pass


class KarolinskaDummyClassifier(Classifier):
    def __init__(self, mask: OpenSlide):
        self.mask = mask

    def classify_patch(self, patch: Patch) -> FeatureSet:
        image = self.mask.read_region(
            location=(patch.x*patch.size.x, patch.y*patch.size.y),
            level=0,
            size=(patch.size.x, patch.size.y))

        r_channel = image.getchannel('R')
        data = np.array(r_channel.getdata())
        feature_set = FeatureSet(patch, {
            "cancer_percentage": sum(map(
                lambda el: 1 if el == 2 else 0, data))/len(data)
        })
        return feature_set
