import unittest
from classifiers import PatchFeature, PatchFeatureCollection,\
    BasicTissueMaskPredictor, TissueClassifier, TissueFeature
from PIL import Image
import numpy as np
from commons import Patch
from test_commons import DummySlide


class DummyModel:
    def __init__(self, res):
        self.res = res

    def predict(self, *args):
        return self.res


class TestTissueDetector(unittest.TestCase):
    def test_detector_no_tissue(self):
        slide_size = (100, 100)
        patch_size = (10, 10)
        image = Image.new('RGB', patch_size)
        slide = DummySlide('slide', slide_size, image)
        model = DummyModel(np.zeros(patch_size[0] * patch_size[1]))
        predictor = BasicTissueMaskPredictor(model)
        tissue_detector = TissueClassifier(predictor)
        patch_collection = tissue_detector.classify(slide, (10, 10))
        self.assertEqual(len(patch_collection), 100)
        for patch_feature in patch_collection:
            self.assertEqual(
                patch_feature.data[TissueFeature.TISSUE_PERCENTAGE], 0)

    def test_detector_all_tissue(self):

        slide_size = (100, 100)
        patch_size = (10, 10)
        image = Image.new('RGB', patch_size)
        slide = DummySlide('slide', slide_size, image)
        model = DummyModel(np.ones(patch_size[0] * patch_size[1]))
        predictor = BasicTissueMaskPredictor(model)
        tissue_detector = TissueClassifier(predictor)
        patch_collection = tissue_detector.classify(slide, (10, 10))
        self.assertEqual(len(patch_collection), 100)
        for patch_feature in patch_collection:
            self.assertEqual(
                patch_feature.data[TissueFeature.TISSUE_PERCENTAGE], 1)


if __name__ == '__main__':
    unittest.main()
