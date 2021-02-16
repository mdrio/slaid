import unittest
from datetime import datetime as dt

import numpy as np
from commons import DummyModel, EddlGreenModel, EddlGreenPatchModel, GreenModel

from slaid.classifiers import BasicClassifier, Filter
from slaid.classifiers.dask import Classifier as DaskClassifier
from slaid.commons import Mask
from slaid.commons.ecvl import load


class BaseTestClassifier:
    LEVEL = 1

    @staticmethod
    def get_classifier(model, feature='tissue'):
        pass

    @staticmethod
    def get_model():
        pass

    def test_classifies_a_slide(self):
        slide = load('tests/data/PH10023-1.thumb.tif')
        model = DummyModel(np.zeros)
        tissue_detector = self.get_classifier(model)
        mask = tissue_detector.classify(slide, level=self.LEVEL)
        self.assertEqual(mask.array.shape,
                         slide.level_dimensions[self.LEVEL][::-1])

    def test_return_all_zeros_if_there_is_no_tissue(self):
        slide = load('tests/data/test.tif')
        model = DummyModel(np.zeros)
        tissue_detector = self.get_classifier(model)
        mask = tissue_detector.classify(slide, level=self.LEVEL)
        self.assertEqual(mask.array.shape[::-1],
                         slide.level_dimensions[self.LEVEL])
        self.assertEqual(mask.array.all(), 0)

    def test_return_all_100_if_all_is_tissue(self):
        slide = load('tests/data/test.tif')
        model = DummyModel(np.ones)
        tissue_detector = self.get_classifier(model)
        mask = tissue_detector.classify(slide, level=self.LEVEL)
        self.assertEqual(mask.array.shape[::-1],
                         slide.level_dimensions[self.LEVEL])
        self.assertTrue((mask.array == 100).all())
        self.assertEqual(mask.array.dtype, 'uint8')

    def test_return_all_1_if_all_is_tissue(self):
        slide = load('tests/data/test.tif')
        model = DummyModel(np.ones)
        tissue_detector = self.get_classifier(model)
        mask = tissue_detector.classify(slide,
                                        level=self.LEVEL,
                                        round_to_0_100=False)
        self.assertEqual(mask.array.shape[::-1],
                         slide.level_dimensions[self.LEVEL])
        self.assertTrue((mask.array == 1).all())
        self.assertEqual(mask.array.dtype, 'float32')

    def test_returns_a_mask(self):
        #  import pudb
        #  pudb.set_trace()
        level = 0
        slide = load('tests/data/test.tif')
        tissue_detector = self.get_classifier(self.get_model())
        mask = tissue_detector.classify(slide, level=level)

        self.assertEqual(mask.array.shape[::-1], slide.level_dimensions[level])
        self.assertEqual(mask.array[:300, :].all(), 1)
        self.assertEqual(mask.array[300:, :].all(), 0)


class BasicClassifierTest(BaseTestClassifier, unittest.TestCase):
    @staticmethod
    def get_classifier(model, feature='tissue'):
        return BasicClassifier(model, feature)

    @staticmethod
    def get_model():
        return GreenModel()


class TestDaskClassifier(BaseTestClassifier, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #  init_client()
        import dask
        dask.config.set(scheduler='synchronous')

    @staticmethod
    def get_classifier(model, feature='tissue'):
        return DaskClassifier(model, feature)

    @staticmethod
    def get_model():
        return GreenModel()


class TestEddlClassifier(BasicClassifierTest):
    @staticmethod
    def get_model():
        return EddlGreenModel()


class TestEddlPatchClassifier(unittest.TestCase):
    @staticmethod
    def get_model(patch_size):
        return EddlGreenPatchModel(patch_size)

    @staticmethod
    def get_classifier(model, feature='tissue'):
        return BasicClassifier(model, feature)

    def test_classifies_by_patch_at_level_0(self, n_batch=1):
        level = 0
        slide = load('tests/data/test.tif')
        print(slide.level_dimensions[level])
        patch_size = (100, 256)
        classifier = self.get_classifier(self.get_model(patch_size))
        mask = classifier.classify(slide, level=level, n_batch=n_batch)

        self.assertEqual(
            mask.array.shape,
            tuple([
                slide.level_dimensions[level][::-1][i] // patch_size[i]
                for i in range(2)
            ]),
        )
        print(mask.array)
        self.assertEqual(mask.array[:3, :].all(), 1)
        self.assertEqual(mask.array[3:, :].all(), 0)

    def test_classifies_with_filter(self):
        level = 0
        patch_size = (100, 100)
        slide = load('tests/data/test.tif')
        mask_array = np.zeros(
            np.array(slide.level_dimensions[level][::-1]) //
            np.array(patch_size))
        mask_array[0, :] = 1
        filter_mask = Mask(mask_array, level, slide.level_downsamples[level],
                           dt.now(), False)

        slide.masks['tissue'] = filter_mask
        classifier = self.get_classifier(self.get_model(patch_size), 'cancer')
        mask = classifier.classify(slide,
                                   level=level,
                                   filter_=Filter(filter_mask) > 0)

        self.assertTrue((mask.array / 100 == filter_mask.array).all())

    def test_classifies_with_no_filtered_patch(self):
        level = 0
        patch_size = (100, 100)
        slide = load('tests/data/test.tif')
        mask_array = np.zeros(
            np.array(slide.level_dimensions[level][::-1]) //
            np.array(patch_size))
        filter_mask = Mask(mask_array, level, slide.level_downsamples[level],
                           dt.now(), False)

        slide.masks['tissue'] = filter_mask
        classifier = self.get_classifier(self.get_model(patch_size), 'cancer')
        mask = classifier.classify(slide,
                                   level=level,
                                   filter_=Filter(filter_mask) > 0)

        self.assertEqual(mask.array.shape, filter_mask.array.shape)
        print(mask.array)
        self.assertTrue(not np.count_nonzero(mask.array))


class TestFilter(unittest.TestCase):
    def test_filters_at_same_level(self):
        array = np.zeros((10, 10))
        indexes_ones = (0, 0)
        array[indexes_ones] = 1
        mask = Mask(array, 0, 1, dt.now(), False)
        filtered = Filter(mask).filter('__gt__', 0.5)

        self.assertTrue((filtered == indexes_ones).all())


if __name__ == '__main__':
    unittest.main()
