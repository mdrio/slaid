import unittest

import numpy as np
from commons import DummyModel, EddlGreenIsTissueModel, GreenIsTissueModel

from slaid.classifiers import BasicClassifier
from slaid.classifiers.dask import RowClassifier
from slaid.commons.ecvl import create_slide

#  from slaid.classifiers.eddl import TissueMaskPredictor as\
#  EddlTissueMaskPredictor


class TestTissueClassifierTest:
    @staticmethod
    def get_classifier(model):
        pass

    @staticmethod
    def get_model():
        pass

    def test_detector_no_tissue(self):
        slide = create_slide('tests/data/test.tif', extraction_level=0)
        model = DummyModel(np.zeros)
        tissue_detector = self.get_classifier(model)
        tissue_detector.classify(slide, include_mask=True)
        self.assertEqual(slide.masks['tissue'].shape[::-1], slide.dimensions)
        for patch in slide.patches:
            self.assertEqual(patch.features['tissue'], 0)
        self.assertEqual(slide.masks['tissue'].all(), 0)

    def test_detector_all_tissue(self):
        slide = create_slide('tests/data/test.tif', extraction_level=0)
        model = DummyModel(np.ones)
        tissue_detector = self.get_classifier(model)
        tissue_detector.classify(slide, include_mask=True)
        self.assertEqual(slide.masks['tissue'].shape[::-1], slide.dimensions)
        for patch in slide.patches:
            self.assertEqual(patch.features['tissue'], 1)

    def test_mask(self):
        slide = create_slide('tests/data/test.tif', extraction_level=0)
        tissue_detector = self.get_classifier(self.get_model())
        tissue_detector.classify(slide, include_mask=True)

        self.assertEqual(slide.masks['tissue'].shape[::-1], slide.dimensions)
        for patch in slide.patches:
            if (patch.y == 0):
                self.assertEqual(patch.features['tissue'], 1)
            else:
                self.assertEqual(patch.features['tissue'], 0)


class BasicTissueClassifierTest(TestTissueClassifierTest, unittest.TestCase):
    @staticmethod
    def get_classifier(model):
        return BasicClassifier(model, 'tissue')

    @staticmethod
    def get_model():
        return GreenIsTissueModel()


class RowClassifierTest(TestTissueClassifierTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #  init_client()
        import dask
        dask.config.set(scheduler='synchronous'
                        )  # overwrite default with single-threaded scheduler

    @staticmethod
    def get_classifier(model):
        return RowClassifier(model, 'tissue', 200)

    @staticmethod
    def get_model():
        return GreenIsTissueModel()


class EddlTissueClassifierTest(BasicTissueClassifierTest):
    @staticmethod
    def get_model():
        return EddlGreenIsTissueModel()


if __name__ == '__main__':
    unittest.main()
