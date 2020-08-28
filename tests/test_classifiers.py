import unittest

import numpy as np
from commons import DummyModel, EddlGreenIsTissueModel, GreenIsTissueModel

from slaid.classifiers import BasicClassifier
from slaid.commons.ecvl import Slide

#  from slaid.classifiers.eddl import TissueMaskPredictor as\
#  EddlTissueMaskPredictor


class TestTissueClassifierTest:
    classifier_cls = None
    predictor_cls = None
    model_cls = None

    def test_detector_no_tissue(self):
        slide = Slide('tests/data/test.tif', extraction_level=0)
        model = DummyModel(np.zeros)
        tissue_detector = self.classifier_cls(model, 'tissue')

        tissue_detector.classify(slide, include_mask=True)
        for patch in slide.patches:
            self.assertEqual(patch.features['tissue'], 0)
        self.assertEqual(slide.masks['tissue'].all(), 0)

    def test_detector_all_tissue(self):
        slide = Slide('tests/data/test.tif', extraction_level=0)
        model = DummyModel(np.ones)
        tissue_detector = self.classifier_cls(model, 'tissue')
        tissue_detector.classify(slide)
        for patch in slide.patches:
            self.assertEqual(patch.features['tissue'], 1)

    def test_mask(self):
        slide = Slide('tests/data/test.tif', extraction_level=0)
        tissue_detector = self.classifier_cls(self.model_cls(), 'tissue')

        tissue_detector.classify(slide, include_mask=True)
        for patch in slide.patches:
            if (patch.y == 0):
                self.assertEqual(patch.features['tissue'], 1)

            else:
                self.assertEqual(patch.features['tissue'], 0)


class BasicTissueClassifierTest(TestTissueClassifierTest, unittest.TestCase):
    classifier_cls = BasicClassifier
    model_cls = GreenIsTissueModel


class EddlTissueClassifierTest(TestTissueClassifierTest, unittest.TestCase):
    classifier_cls = BasicClassifier
    model_cls = EddlGreenIsTissueModel


if __name__ == '__main__':
    unittest.main()
