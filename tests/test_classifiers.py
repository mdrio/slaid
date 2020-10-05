import unittest

import numpy as np
from commons import DummyModel, EddlGreenIsTissueModel, GreenIsTissueModel

from slaid.classifiers import BasicClassifier
from slaid.classifiers.dask import Classifier as DaskClassifier
from slaid.commons import Mask, Patch, convert_patch
from slaid.commons.ecvl import create_slide


class TestTissueClassifierTest:
    LEVEL = 1

    @staticmethod
    def get_classifier(model, feature='tissue'):
        pass

    @staticmethod
    def get_model():
        pass

    def test_detector_no_tissue(self):
        slide = create_slide('tests/data/test.tif')
        model = DummyModel(np.zeros)
        tissue_detector = self.get_classifier(model)
        mask = tissue_detector.classify(slide, level=self.LEVEL)
        self.assertEqual(mask.array.shape[::-1],
                         slide.level_dimensions[self.LEVEL])
        self.assertEqual(mask.array.all(), 0)

    def test_detector_all_tissue(self):
        slide = create_slide('tests/data/test.tif')
        model = DummyModel(np.ones)
        tissue_detector = self.get_classifier(model)
        mask = tissue_detector.classify(slide, level=self.LEVEL)
        self.assertEqual(mask.array.shape[::-1],
                         slide.level_dimensions[self.LEVEL])

    def test_mask(self):
        level = 0
        slide = create_slide('tests/data/test.tif')
        tissue_detector = self.get_classifier(self.get_model())
        mask = tissue_detector.classify(slide, level=level)

        self.assertEqual(mask.array.shape[::-1], slide.level_dimensions[level])
        self.assertEqual(mask.array[:300, :].all(), 1)
        self.assertEqual(mask.array[300:, :].all(), 0)

    def test_classify_by_patch_level_0(self):
        level = 0
        slide = create_slide('tests/data/test.tif')
        tissue_detector = self.get_classifier(self.get_model())
        mask = tissue_detector.classify(slide,
                                        level=level,
                                        patch_size=(200, 200))

        self.assertEqual(mask.array.shape[::-1], slide.level_dimensions[level])
        self.assertEqual(mask.array[:300, :].all(), 1)
        self.assertEqual(mask.array[300:, :].all(), 0)

    def test_classify_by_patch_level_2(self):
        level = 2
        slide = create_slide('tests/data/test.tif')
        downsample = slide.level_downsamples[level]
        tissue_detector = self.get_classifier(self.get_model())
        mask = tissue_detector.classify(slide,
                                        level=level,
                                        patch_size=(200, 200))

        self.assertEqual(mask.array.shape[::-1], slide.level_dimensions[level])
        self.assertEqual(mask.array[:round(300 // downsample), :].all(), 1)
        self.assertEqual(mask.array[round(300 // downsample):, :].all(), 0)

    def test_classify_with_filter_same_level(self):
        tissue_level = 0
        cancer_level = 0
        patch_size = (100, 100)
        slide = create_slide('tests/data/test.tif')
        mask_array = np.zeros(slide.level_dimensions[tissue_level][::-1])
        mask = Mask(mask_array, tissue_level,
                    slide.level_downsamples[tissue_level])
        tissue_patch = Patch(100, 100, patch_size,
                             slide.level_downsamples[tissue_level])
        cancer_patch = convert_patch(tissue_patch, slide,
                                     slide.level_downsamples[cancer_level])
        mask_array[tissue_patch.y:tissue_patch.y + tissue_patch.size[1],
                   tissue_patch.x:tissue_patch.x +
                   tissue_patch.size[0]] = np.ones(tissue_patch.size)

        slide.masks['tissue'] = mask
        tissue_detector = self.get_classifier(self.get_model(), 'cancer')
        #  import pudb
        #  pudb.set_trace()
        mask = tissue_detector.classify(slide,
                                        level=cancer_level,
                                        patch_size=patch_size,
                                        patch_filter='tissue >= 1')

        self.assertEqual(mask.array.shape[::-1],
                         slide.level_dimensions[cancer_level])
        self.assertEqual(mask.ratio(cancer_patch), 1)
        self.assertEqual(np.sum(mask.array), cancer_patch.area)

    def test_classify_with_filter_different_level_proportional_patch_size(
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
        cancer_patch = convert_patch(tissue_patch, slide,
                                     slide.level_downsamples[cancer_level])
        mask_array[tissue_patch.y:tissue_patch.y + tissue_patch.size[1],
                   tissue_patch.x:tissue_patch.x +
                   tissue_patch.size[0]] = np.ones(tissue_patch.size)

        slide.masks['tissue'] = mask
        downsample = slide.level_downsamples[cancer_level]
        tissue_detector = self.get_classifier(self.get_model(), 'cancer')
        mask = tissue_detector.classify(
            slide,
            level=cancer_level,
            patch_size=(int(patch_size[0] // downsample),
                        int(patch_size[1] // downsample)),
            patch_filter='tissue >= 1')

        self.assertEqual(mask.array.shape[::-1],
                         slide.level_dimensions[cancer_level])
        self.assertEqual(mask.ratio(tissue_patch), 1)
        self.assertEqual(np.sum(mask.array), cancer_patch.area)

    def test_classify_with_filter_different_level_not_proportional_patch_size(
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

        mask_array[tissue_patch.y:tissue_patch.y + tissue_patch.size[1],
                   tissue_patch.x:tissue_patch.x +
                   tissue_patch.size[0]] = np.ones(tissue_patch.size)

        slide.masks['tissue'] = mask

        downsample = slide.level_downsamples[cancer_level]
        expected_ratio = patch_size[0]**2 / (downsample * patch_size[0])**2

        tissue_detector = self.get_classifier(self.get_model(), 'cancer')
        mask = tissue_detector.classify(
            slide,
            level=cancer_level,
            patch_size=patch_size,
            patch_filter=f'tissue >= {expected_ratio}')

        self.assertEqual(mask.array.shape[::-1],
                         slide.level_dimensions[cancer_level])
        self.assertEqual(np.sum(mask.array), patch_size[0]**2)


class BasicClassifierTest(TestTissueClassifierTest, unittest.TestCase):
    @staticmethod
    def get_classifier(model, feature='tissue'):
        return BasicClassifier(model, feature)

    @staticmethod
    def get_model():
        return GreenIsTissueModel()


class DaskClassifierTest(TestTissueClassifierTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #  init_client()
        import dask
        dask.config.set(scheduler='synchronous'
                        )  # overwrite default with single-threaded scheduler

    @staticmethod
    def get_classifier(model, feature='tissue'):
        return DaskClassifier(model, feature)

    @staticmethod
    def get_model():
        return GreenIsTissueModel()


class EddlTissueClassifierTest(BasicClassifierTest):
    @staticmethod
    def get_model():
        return EddlGreenIsTissueModel()


if __name__ == '__main__':
    unittest.main()
