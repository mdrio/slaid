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
    def setUp(self):
        self.slide_size = (200, 100)
        self.slide = DummySlide('slide', self.slide_size)
        self.patch_size = (10, 10)
        self.collection = PandasPatchCollection(self.slide, self.patch_size)

    def test_init(self):
        self.assertEqual(
            len(self.collection), self.slide_size[0] * self.slide_size[1] /
            (self.patch_size[0] * self.patch_size[1]))

    def test_iteration(self):
        x = y = counter = 0
        for patch in self.collection:
            self.assertEqual(patch.x, x)
            self.assertEqual(patch.y, y)
            x = (x + self.patch_size[0]) % self.slide_size[0]
            if x == 0:
                y += self.patch_size[1]
            counter += 1

        self.assertEqual(
            counter, self.slide_size[0] * self.slide_size[1] /
            (self.patch_size[0] * self.patch_size[1]))

    def test_get_item(self):
        coordinates = (190, 90)
        patch = self.collection.get_patch(coordinates)
        self.assertTrue(isinstance(patch, Patch))
        self.assertEqual(patch.x, coordinates[0])
        self.assertEqual(patch.y, coordinates[1])

    def test_update_patch(self):
        coordinates = (190, 90)
        self.collection.update_patch(coordinates=coordinates,
                                     features={
                                         'test': 1,
                                         'test2': 2
                                     })
        self.assertEqual(len(self.collection), 200)
        patch = self.collection.get_patch(coordinates)
        self.assertEqual(patch.x, coordinates[0])
        self.assertEqual(patch.y, coordinates[1])
        self.assertEqual(patch.features['test'], 1)
        self.assertEqual(patch.features['test2'], 2)

    def test_filter(self):
        for i, p in enumerate(self.collection):
            self.collection.update_patch(patch=p, features={'feature': i})
        filtered_collection = self.collection.loc[
            self.collection['feature'] > 0]
        self.assertEqual(len(filtered_collection), len(self.collection) - 1)

        filtered_collection.update_patch(coordinates=(10, 10),
                                         features={
                                             'feature2': -1,
                                         })
        #  self.collection.update(filtered_collection)
        #  self.assertEqual(
        #      self.collection.get_patch((10, 10)).features['feature'], -1)

        self.collection.merge(filtered_collection)
        self.assertEqual(
            self.collection.get_patch((10, 10)).features['feature2'], -1)


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
