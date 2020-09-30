import unittest

import numpy as np
from commons import DummyModel, EddlGreenIsTissueModel, GreenIsTissueModel

from slaid.classifiers import BasicClassifier
from slaid.commons.ecvl import create_slide

#  from slaid.classifiers.eddl import TissueMaskPredictor as\
#  EddlTissueMaskPredictor


class TestTissueClassifierTest:
    LEVEL = 1

    @staticmethod
    def get_classifier(model):
        pass

    @staticmethod
    def get_model():
        pass

    def test_detector_no_tissue(self):
        slide = create_slide('tests/data/test.tif')
        model = DummyModel(np.zeros)
        tissue_detector = self.get_classifier(model)
        tissue_detector.classify(slide, level=self.LEVEL)
        self.assertEqual(slide.masks['tissue'].array.shape[::-1],
                         slide.level_dimensions[self.LEVEL])
        self.assertEqual(slide.masks['tissue'].array.all(), 0)

    def test_detector_all_tissue(self):
        slide = create_slide('tests/data/test.tif')
        model = DummyModel(np.ones)
        tissue_detector = self.get_classifier(model)
        tissue_detector.classify(slide, level=self.LEVEL)
        self.assertEqual(slide.masks['tissue'].array.shape[::-1],
                         slide.level_dimensions[self.LEVEL])

    def test_mask(self):
        level = 0
        slide = create_slide('tests/data/test.tif')
        tissue_detector = self.get_classifier(self.get_model())
        tissue_detector.classify(slide, level=level)

        mask = slide.masks['tissue'].array
        self.assertEqual(mask.shape[::-1], slide.level_dimensions[level])
        self.assertEqual(mask[:300, :].all(), 1)
        self.assertEqual(mask[300:, :].all(), 0)

    def test_classify_by_patch(self):
        level = 0
        slide = create_slide('tests/data/test.tif')
        tissue_detector = self.get_classifier(self.get_model())
        tissue_detector.classify(slide, level=level, patch_size=(200, 200))

        mask = slide.masks['tissue'].array
        self.assertEqual(mask.shape[::-1], slide.level_dimensions[level])
        self.assertEqual(mask[:300, :].all(), 1)
        self.assertEqual(mask[300:, :].all(), 0)


class BasicTissueClassifierTest(TestTissueClassifierTest, unittest.TestCase):
    @staticmethod
    def get_classifier(model):
        return BasicClassifier(model, 'tissue')

    @staticmethod
    def get_model():
        return GreenIsTissueModel()


#  class RowClassifierTest(TestTissueClassifierTest, unittest.TestCase):
#      @classmethod
#      def setUpClass(cls):
#          #  init_client()
#          import dask
#          dask.config.set(scheduler='synchronous'
#                          )  # overwrite default with single-threaded scheduler
#
#      @staticmethod
#      def get_classifier(model):
#          return RowClassifier(model, 'tissue', 200)
#
#      @staticmethod
#      def get_model():
#          return GreenIsTissueModel()
#


class EddlTissueClassifierTest(BasicTissueClassifierTest):
    @staticmethod
    def get_model():
        return EddlGreenIsTissueModel()


if __name__ == '__main__':
    unittest.main()
