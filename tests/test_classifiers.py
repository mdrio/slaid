import unittest
from commons import Patch
from classifiers import BasicTissueMaskPredictor,\
    TissueClassifier, TissueFeature, PandasPatchCollection,\
    KarolinskaTrueValueClassifier, KarolinskaFeature
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
    def test_init(self):
        slide_size = (100, 100)
        patch_size = (10, 10)
        #  slide_size = (23904, 28664)
        #  patch_size = (256, 256)
        slide = DummySlide('slide', slide_size)
        collection = PandasPatchCollection(slide, patch_size)
        self.assertEqual(
            len(collection),
            slide_size[0] * slide_size[1] / (patch_size[0] * patch_size[1]))

    def test_iteration(self):
        slide_size = (200, 100)
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

        self.assertEqual(
            counter,
            slide_size[0] * slide_size[1] / (patch_size[0] * patch_size[1]))

    def test_get_item(self):
        slide = DummySlide('slide', (200, 100))
        patch_size = (10, 10)
        coordinates = (190, 90)
        collection = PandasPatchCollection(slide, patch_size)
        patch = collection[coordinates]
        self.assertTrue(isinstance(patch, Patch))
        self.assertEqual(patch.x, coordinates[0])
        self.assertEqual(patch.y, coordinates[1])

    def test_update_patch(self):
        slide_size = (200, 100)
        slide = DummySlide('slide', slide_size)
        patch_size = (10, 10)
        collection = PandasPatchCollection(slide, patch_size)
        coordinates = (190, 90)
        collection.update_patch(coordinates=coordinates, features={'test': 1})
        self.assertEqual(len(collection), 200)
        patch = collection[coordinates]
        self.assertEqual(patch.x, coordinates[0])
        self.assertEqual(patch.y, coordinates[1])
        self.assertEqual(patch.features['test'], 1)


class KarolinskaTest(unittest.TestCase):
    def test_true_value(self):
        size = (200, 100)
        patch_size = (10, 10)
        #  size = (23904, 28664)
        #  patch_size = (256, 256)
        data = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        data[0:10, 0:50] = [2, 0, 0]

        mask_slide = DummySlide('mask', size, data=data)
        slide = DummySlide('slide', size)
        cl = KarolinskaTrueValueClassifier(mask_slide)
        collection = PandasPatchCollection(slide, patch_size)
        collection_classified = cl.classify(collection)
        self.assertEqual(len(collection), len(collection_classified))
        for i, patch in enumerate(collection_classified):
            feature = patch.features[KarolinskaFeature.CANCER_PERCENTAGE]
            if i <= 4:
                self.assertEqual(feature, 1)
            else:
                self.assertEqual(feature, 0)


if __name__ == '__main__':
    unittest.main()
