import unittest
from collections import defaultdict

import numpy as np
from commons import DummyModel, EddlGreenIsTissueModel, GreenIsTissueModel

from slaid.classifiers import BasicClassifier, Batch, Filter, Patch
from slaid.classifiers.dask import Classifier as DaskClassifier
from slaid.commons import Mask
from slaid.commons.ecvl import create_slide


class BaseTestClassifier:
    LEVEL = 1

    @staticmethod
    def get_classifier(model, feature='tissue'):
        pass

    @staticmethod
    def get_model():
        pass

    def test_classifies_a_slide(self):
        slide = create_slide('tests/data/PH10023-1.thumb.tif')
        model = DummyModel(np.zeros)
        tissue_detector = self.get_classifier(model)
        #  import pudb
        #  pudb.set_trace()
        mask = tissue_detector.classify(slide, level=self.LEVEL)
        self.assertEqual(mask.array.shape,
                         slide.level_dimensions[self.LEVEL][::-1])

    def test_return_all_zeros_if_there_is_no_tissue(self):
        slide = create_slide('tests/data/test.tif')
        model = DummyModel(np.zeros)
        tissue_detector = self.get_classifier(model)
        mask = tissue_detector.classify(slide, level=self.LEVEL)
        self.assertEqual(mask.array.shape[::-1],
                         slide.level_dimensions[self.LEVEL])
        self.assertEqual(mask.array.all(), 0)

    def test_return_all_ones_if_all_is_tissue(self):
        slide = create_slide('tests/data/test.tif')
        model = DummyModel(np.ones)
        tissue_detector = self.get_classifier(model)
        mask = tissue_detector.classify(slide, level=self.LEVEL)
        self.assertEqual(mask.array.shape[::-1],
                         slide.level_dimensions[self.LEVEL])

    def test_returns_a_mask(self):
        level = 0
        slide = create_slide('tests/data/test.tif')
        tissue_detector = self.get_classifier(self.get_model())
        mask = tissue_detector.classify(slide, level=level)

        self.assertEqual(mask.array.shape[::-1], slide.level_dimensions[level])
        self.assertEqual(mask.array[:300, :].all(), 1)
        self.assertEqual(mask.array[300:, :].all(), 0)

    def test_classifies_by_patch_at_level_0(self, n_batch=1):
        level = 0
        slide = create_slide('tests/data/test.tif')
        tissue_detector = self.get_classifier(self.get_model())
        mask = tissue_detector.classify(slide,
                                        level=level,
                                        patch_size=(200, 200),
                                        n_batch=n_batch)

        self.assertEqual(mask.array.shape[::-1], slide.level_dimensions[level])
        self.assertEqual(mask.array[:300, :].all(), 1)
        self.assertEqual(mask.array[300:, :].all(), 0)

    def test_classifies_by_patch_at_level_0_with_2_batches(self):
        self.test_classifies_by_patch_at_level_0(2)

    def test_classifies_by_patch_at_level_1(self):
        level = 1
        slide = create_slide('tests/data/test.tif')
        downsample = slide.level_downsamples[level]
        tissue_detector = self.get_classifier(self.get_model())

        mask = tissue_detector.classify(slide,
                                        level=level,
                                        patch_size=(200, 200))

        self.assertEqual(mask.array.shape[::-1], slide.level_dimensions[level])
        self.assertEqual(mask.array[:round(300 // downsample), :].all(), 1)
        self.assertEqual(mask.array[round(300 // downsample):, :].all(), 0)

    def test_classifies_with_filter_at_the_same_level(self):
        tissue_level = 0
        cancer_level = 0
        patch_size = (100, 100)
        slide = create_slide('tests/data/test.tif')
        mask_array = np.zeros(slide.level_dimensions[tissue_level][::-1])
        mask = Mask(mask_array, tissue_level,
                    slide.level_downsamples[tissue_level])
        tissue_patch = Patch(100, 100, patch_size,
                             slide.level_downsamples[tissue_level])
        mask_array[tissue_patch.row:tissue_patch.row + tissue_patch.size[0],
                   tissue_patch.col:tissue_patch.col +
                   tissue_patch.size[1]] = np.ones(tissue_patch.size)

        slide.masks['tissue'] = mask
        tissue_detector = self.get_classifier(self.get_model(), 'cancer')
        mask = tissue_detector.classify(slide,
                                        level=cancer_level,
                                        patch_size=patch_size,
                                        filter_=Filter.create(
                                            slide, 'tissue >= 1'))

        self.assertEqual(mask.array.shape[::-1],
                         slide.level_dimensions[cancer_level])
        self.assertEqual(np.sum(mask.array), 10000)

    def test_classifies_with_filter_at_a_different_level_with_proportional_patch_size(
            self):
        tissue_level = 0
        cancer_level = 2
        patch_size = (100, 100)
        slide = create_slide('tests/data/test.tif')
        mask_array = np.zeros(slide.level_dimensions[tissue_level][::-1])
        mask = Mask(mask_array, tissue_level,
                    slide.level_downsamples[tissue_level])
        tissue_patch = Patch(100, 100, patch_size,
                             slide.level_downsamples[tissue_level])
        mask_array[tissue_patch.row:tissue_patch.row + tissue_patch.size[0],
                   tissue_patch.col:tissue_patch.col +
                   tissue_patch.size[1]] = np.ones(tissue_patch.size)

        slide.masks['tissue'] = mask
        downsample = slide.level_downsamples[cancer_level]
        tissue_detector = self.get_classifier(self.get_model(), 'cancer')
        mask = tissue_detector.classify(
            slide,
            level=cancer_level,
            patch_size=(int(patch_size[0] // downsample),
                        int(patch_size[1] // downsample)),
            filter_=Filter.create(slide, 'tissue >= 1'))

        self.assertEqual(mask.array.shape[::-1],
                         slide.level_dimensions[cancer_level])
        self.assertTrue(np.sum(mask.array) > 0)

    def test_classifies_filter_at_different_level__not_proportional_patch_size(
            self):
        tissue_level = 0
        cancer_level = 2
        patch_size = (75, 75)
        slide = create_slide('tests/data/test.tif')
        mask_array = np.zeros(slide.level_dimensions[tissue_level][::-1])
        mask = Mask(mask_array, tissue_level,
                    slide.level_downsamples[tissue_level])
        tissue_patch = Patch(100, 100, patch_size,
                             slide.level_downsamples[tissue_level])

        mask_array[tissue_patch.row:tissue_patch.row + tissue_patch.size[0],
                   tissue_patch.col:tissue_patch.col +
                   tissue_patch.size[1]] = np.ones(tissue_patch.size)

        slide.masks['tissue'] = mask

        downsample = slide.level_downsamples[cancer_level]
        expected_ratio = patch_size[0]**2 / (downsample * patch_size[0])**2

        tissue_detector = self.get_classifier(self.get_model(), 'cancer')
        mask = tissue_detector.classify(slide,
                                        level=cancer_level,
                                        patch_size=patch_size,
                                        filter_=Filter.create(
                                            slide,
                                            f'tissue >= {expected_ratio}'))

        self.assertEqual(mask.array.shape[::-1],
                         slide.level_dimensions[cancer_level])
        self.assertTrue(np.sum(mask.array) > 0)


class BasicClassifierTest(BaseTestClassifier, unittest.TestCase):
    @staticmethod
    def get_classifier(model, feature='tissue'):
        return BasicClassifier(model, feature)

    @staticmethod
    def get_model():
        return GreenIsTissueModel()


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
        return GreenIsTissueModel()


class TestEddlClassifier(BasicClassifierTest):
    @staticmethod
    def get_model():
        return EddlGreenIsTissueModel()


class TestBatch(unittest.TestCase):
    def test_produces_array_with_a_correct_shape(self):
        slide = create_slide('tests/data/PH10023-1.thumb.tif')
        classifier = BasicClassifier(DummyModel(np.zeros), 'tissue')
        n_batches = [1, 2, 3, 5, 10, 100]
        patch_sizes = [(256, 256), (256, 256), None, None, None, None]
        level = 0
        for n_batch, patch_size in zip(n_batches, patch_sizes):
            batches = list(
                Batch(start, size, np.zeros(size[::-1]), 1)
                for start, size in classifier._get_batch_coordinates(
                    slide, level, n_batch, patch_size))
            batches_array = np.concatenate([b.array for b in batches], axis=0)
            self.assertEqual(slide.level_dimensions[level],
                             batches_array.shape[:2][::-1])

    def test_returns_correct_patches(self):
        slide = create_slide('tests/data/PH10023-1.thumb.tif')
        classifier = BasicClassifier(DummyModel(np.zeros), 'tissue')
        n_batch = 1
        patch_size = (256, 256)
        level = 0
        batches = list(
            Batch(start, size, np.zeros(size[::-1]), 1)
            for start, size in classifier._get_batch_coordinates(
                slide, level, n_batch, patch_size))

        rows = []
        for b in batches:
            patches_by_row = defaultdict(list)
            for p in b.get_patches(patch_size):
                patches_by_row[p.row].append(p)
            for r in patches_by_row.values():
                rows.append(np.concatenate([_.array for _ in r], axis=1))
        mask = np.concatenate(rows, axis=0)
        self.assertEqual(mask.shape[:2], slide.level_dimensions[level][::-1])


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
