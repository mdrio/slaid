import unittest
from collections import defaultdict

import numpy as np
from commons import DummyModel, EddlGreenModel, GreenModel, EddlGreenPatchModel

from slaid.classifiers import BasicClassifier, Batch, Filter, Patch
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

    def test_return_all_ones_if_all_is_tissue(self):
        slide = load('tests/data/test.tif')
        model = DummyModel(np.ones)
        tissue_detector = self.get_classifier(model)
        mask = tissue_detector.classify(slide, level=self.LEVEL)
        self.assertEqual(mask.array.shape[::-1],
                         slide.level_dimensions[self.LEVEL])

    def test_returns_a_mask(self):
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

    #  def test_classifies_by_patch_at_level_0_with_2_batches(self):
    #      self.test_classifies_by_patch_at_level_0(2)
    #
    #  def test_classifies_by_patch_at_level_1(self):
    #      level = 1
    #      slide = load('tests/data/test.tif')
    #      downsample = slide.level_downsamples[level]
    #      tissue_detector = self.get_classifier(self.get_model())
    #
    #      mask = tissue_detector.classify(slide,
    #                                      level=level,
    #                                      patch_size=(200, 200))
    #
    #      self.assertEqual(mask.array.shape[::-1], slide.level_dimensions[level])
    #      self.assertEqual(mask.array[:round(300 // downsample), :].all(), 1)
    #      self.assertEqual(mask.array[round(300 // downsample):, :].all(), 0)
    #
    #  def test_classifies_with_filter_at_the_same_level(self):
    #      tissue_level = 0
    #      cancer_level = 0
    #      patch_size = (100, 100)
    #      slide = load('tests/data/test.tif')
    #      mask_array = np.zeros(slide.level_dimensions[tissue_level][::-1])
    #      mask = Mask(mask_array, tissue_level,
    #                  slide.level_downsamples[tissue_level])
    #      tissue_patch = Patch(100, 100, patch_size,
    #                           slide.level_downsamples[tissue_level])
    #      mask_array[tissue_patch.row:tissue_patch.row + tissue_patch.size[0],
    #                 tissue_patch.col:tissue_patch.col +
    #                 tissue_patch.size[1]] = np.ones(tissue_patch.size)
    #
    #      slide.masks['tissue'] = mask
    #      tissue_detector = self.get_classifier(self.get_model(), 'cancer')
    #      mask = tissue_detector.classify(slide,
    #                                      level=cancer_level,
    #                                      patch_size=patch_size,
    #                                      filter_=Filter.create(
    #                                          slide, 'tissue >= 1'))
    #
    #      self.assertEqual(mask.array.shape[::-1],
    #                       slide.level_dimensions[cancer_level])
    #      self.assertEqual(np.sum(mask.array), 10000)
    #
    #  def test_classifies_with_filter_at_a_different_level_with_proportional_patch_size(
    #          self):
    #      tissue_level = 0
    #      cancer_level = 2
    #      patch_size = (100, 100)
    #      slide = load('tests/data/test.tif')
    #      mask_array = np.zeros(slide.level_dimensions[tissue_level][::-1])
    #      mask = Mask(mask_array, tissue_level,
    #                  slide.level_downsamples[tissue_level])
    #      tissue_patch = Patch(100, 100, patch_size,
    #                           slide.level_downsamples[tissue_level])
    #      mask_array[tissue_patch.row:tissue_patch.row + tissue_patch.size[0],
    #                 tissue_patch.col:tissue_patch.col +
    #                 tissue_patch.size[1]] = np.ones(tissue_patch.size)
    #
    #      slide.masks['tissue'] = mask
    #      downsample = slide.level_downsamples[cancer_level]
    #      tissue_detector = self.get_classifier(self.get_model(), 'cancer')
    #      mask = tissue_detector.classify(
    #          slide,
    #          level=cancer_level,
    #          patch_size=(int(patch_size[0] // downsample),
    #                      int(patch_size[1] // downsample)),
    #          filter_=Filter.create(slide, 'tissue >= 1'))
    #
    #      self.assertEqual(mask.array.shape[::-1],
    #                       slide.level_dimensions[cancer_level])
    #      self.assertTrue(np.sum(mask.array) > 0)
    #
    #  def test_classifies_filter_at_different_level__not_proportional_patch_size(
    #          self):
    #      tissue_level = 0
    #      cancer_level = 2
    #      patch_size = (75, 75)
    #      slide = load('tests/data/test.tif')
    #      mask_array = np.zeros(slide.level_dimensions[tissue_level][::-1])
    #      mask = Mask(mask_array, tissue_level,
    #                  slide.level_downsamples[tissue_level])
    #      tissue_patch = Patch(100, 100, patch_size,
    #                           slide.level_downsamples[tissue_level])
    #
    #      mask_array[tissue_patch.row:tissue_patch.row + tissue_patch.size[0],
    #                 tissue_patch.col:tissue_patch.col +
    #                 tissue_patch.size[1]] = np.ones(tissue_patch.size)
    #
    #      slide.masks['tissue'] = mask
    #
    #      downsample = slide.level_downsamples[cancer_level]
    #      expected_ratio = patch_size[0]**2 / (downsample * patch_size[0])**2
    #
    #      tissue_detector = self.get_classifier(self.get_model(), 'cancer')
    #      mask = tissue_detector.classify(slide,
    #                                      level=cancer_level,
    #                                      patch_size=patch_size,
    #                                      filter_=Filter.create(
    #                                          slide,
    #                                          f'tissue >= {expected_ratio}'))
    #
    #      self.assertEqual(mask.array.shape[::-1],
    #                       slide.level_dimensions[cancer_level])
    #      self.assertTrue(np.sum(mask.array) > 0)
    #


class TestFilter(unittest.TestCase):
    def test_filters_at_same_level(self):
        array = np.zeros((10, 10))
        array[0, 0] = 1
        mask = Mask(array, 0, 1)

        batch = Batch((0, 0), (2, 10), np.zeros((2, 10)), 1)

        filter_ = Filter(mask, '__gt__', 0.5)
        filtered = filter_.filter(batch)

        expected = np.zeros((2, 10), dtype=bool)
        expected[0, 0] = True
        self.assertTrue((filtered == expected).all())

    def test_filters_at_different_level(self):
        array = np.zeros((10, 10))
        array[0, 0] = 1
        mask = Mask(array, 1, 2)

        batch = Batch((0, 0), (2, 20), np.zeros((2, 20)), 1)

        filter_ = Filter(mask, '__gt__', 0.5)
        filtered = filter_.filter(batch)

        expected = np.zeros((2, 20), dtype=bool)
        expected[0, 0] = True
        expected[0, 1] = True
        expected[1, 0] = True
        self.assertTrue((filtered == expected).all())


if __name__ == '__main__':
    unittest.main()
