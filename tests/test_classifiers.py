import unittest
from commons import Patch
from classifiers import BasicTissueMaskPredictor,\
    TissueClassifier, TissueFeature, PandasPatchCollection
import numpy as np
from test_commons import DummySlide


class DummyModel:
    def __init__(self, func):
        self.func = func

    def predict(self, array):
        return self.func(array.shape[0])


class TestTissueDetector(unittest.TestCase):
    def test_detector_no_tissue(self):
        slide = DummySlide('slide', (100, 100))
        model = DummyModel(np.zeros)
        tissue_detector = TissueClassifier(BasicTissueMaskPredictor(model))

        patch_size = (10, 10)
        patch_collection = tissue_detector.classify(
            PandasPatchCollection(slide, patch_size))
        self.assertEqual(len(patch_collection), patch_size[0] * patch_size[1])
        for patch in patch_collection:
            self.assertEqual(patch.features[TissueFeature.TISSUE_PERCENTAGE],
                             0)

    def test_detector_all_tissue(self):
        slide = DummySlide('slide', (100, 100))
        model = DummyModel(np.ones)
        tissue_detector = TissueClassifier(BasicTissueMaskPredictor(model))

        patch_size = (10, 10)
        patch_collection = tissue_detector.classify(
            PandasPatchCollection(slide, patch_size))
        self.assertEqual(len(patch_collection), patch_size[0] * patch_size[1])
        for patch_feature in patch_collection:
            self.assertEqual(
                patch_feature.features[TissueFeature.TISSUE_PERCENTAGE], 1)


class TestPandasPatchCollection(unittest.TestCase):
    def test_iteration_order(self):
        slide_size = (100, 100)
        slide = DummySlide('slide', slide_size)
        patch_size = (10, 10)
        collection = PandasPatchCollection(slide, patch_size)
        x = y = counter = 0
        for patch in collection:
            self.assertEqual(patch.x, x)
            self.assertEqual(patch.y, y)
            x = (x + patch_size[0]) % slide_size[0]
            if x == 0:
                y += patch_size[1]
            counter += 1

        self.assertEqual(counter, 100)

    def test_get_item(self):
        slide = DummySlide('slide', (100, 100))
        patch_size = (10, 10)
        collection = PandasPatchCollection(slide, patch_size)
        patch = collection[(10, 20)]
        self.assertTrue(isinstance(patch, Patch))
        self.assertEqual(patch.x, 10)
        self.assertEqual(patch.y, 20)


#

if __name__ == '__main__':
    unittest.main()
