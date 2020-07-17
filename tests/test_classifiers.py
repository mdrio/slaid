import unittest
from classifiers import BasicTissueMaskPredictor,\
    TissueClassifier, TissueFeature
from PIL import Image
import numpy as np
from test_commons import DummySlide


class DummyModel:
    def __init__(self, func):
        self.func = func

    def predict(self, array):
        return self.func(array.shape[0])


class TestTissueDetector(unittest.TestCase):
    def test_detector_no_tissue(self):
        slide_size = (100, 100)
        slide = DummySlide('slide', slide_size)
        model = DummyModel(np.zeros)
        predictor = BasicTissueMaskPredictor(model)
        tissue_detector = TissueClassifier(predictor)
        patch_collection = tissue_detector.classify(slide, (10, 10))
        self.assertEqual(len(patch_collection), 100)
        for patch_feature in patch_collection:
            self.assertEqual(
                patch_feature.data[TissueFeature.TISSUE_PERCENTAGE], 0)

    def test_detector_all_tissue(self):

        slide_size = (100, 100)
        slide = DummySlide('slide', slide_size)
        model = DummyModel(np.ones)
        predictor = BasicTissueMaskPredictor(model)
        tissue_detector = TissueClassifier(predictor)
        patch_collection = tissue_detector.classify(slide, (10, 10))
        self.assertEqual(len(patch_collection), 100)
        for patch_feature in patch_collection:
            self.assertEqual(
                patch_feature.data[TissueFeature.TISSUE_PERCENTAGE], 1)


if __name__ == '__main__':
    unittest.main()
