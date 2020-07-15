import unittest
from classifiers import PatchFeature, PatchFeatureCollection,\
    BasicTissueMaskPredictor, TissueDetector
from PIL import Image
import numpy as np
from commons import Patch
from test_commons import DummySlide


class DummyModel:
    def __init__(self, res):
        self.res = res

    def get_predict(self, *args):
        return self.res


class TestPatchFeatureCollection(unittest.TestCase):
    def test_sort(self):
        slide = DummySlide('slide', (400, 200))
        patch_size = (100, 100)
        features = [
            PatchFeature(Patch(slide, (0, 0), patch_size), {}),
            PatchFeature(Patch(slide, (0, 100), patch_size), {}),
            PatchFeature(Patch(slide, (0, 200), patch_size), {}),
            PatchFeature(Patch(slide, (0, 300), patch_size), {}),
            PatchFeature(Patch(slide, (
                0,
                0,
            ), patch_size), {}),
            PatchFeature(Patch(slide, (
                0,
                100,
            ), patch_size), {}),
            PatchFeature(Patch(slide, (
                0,
                200,
            ), patch_size), {}),
            PatchFeature(Patch(slide, (
                0,
                300,
            ), patch_size), {}),
        ]

        reversed_features = list(reversed(features))
        collection = PatchFeatureCollection(slide, patch_size,
                                            reversed_features)
        collection.sort()
        self.assertEqual(features, collection.features)


class TestTissueDetector(unittest.TestCase):
    def test_detector_no_tissue(self):
        size = (100, 100)
        image = Image.new('RGB', size)
        slide = DummySlide('slide', size, image)
        model = DummyModel(np.zeros(size[0] * size[1]))
        predictor = BasicTissueMaskPredictor(model)
        tissue_detector = TissueDetector(predictor)
        patches = tissue_detector.extract_patches(slide, (10, 10))


if __name__ == '__main__':
    unittest.main()
