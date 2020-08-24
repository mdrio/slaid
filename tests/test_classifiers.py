import unittest

import numpy as np
from commons import DummyModel, DummySlide, GreenIsTissueModel, EddlGreenIsTissueModel

from slaid.classifiers import (BasicTissueClassifier, BasicTissueMaskPredictor,
                               InterpolatedTissueClassifier, KarolinskaFeature,
                               KarolinskaTrueValueClassifier, TissueFeature)
from slaid.commons.ecvl import Slide

#  from slaid.classifiers.eddl import TissueMaskPredictor as\
#  EddlTissueMaskPredictor


class TestTissueClassifierTest:
    classifier_cls = None
    predictor_cls = None
    model_cls = None

    def test_detector_no_tissue(self):
        slide = Slide('data/test.tif', extraction_level=0)
        model = DummyModel(np.zeros)
        tissue_detector = self.classifier_cls(self.predictor_cls(model))

        tissue_detector.classify(slide)
        for patch in slide.patches:
            self.assertEqual(patch.features[TissueFeature.TISSUE_PERCENTAGE],
                             0)

    def test_detector_all_tissue(self):
        slide = Slide('data/test.tif', extraction_level=0)
        model = DummyModel(np.ones)
        tissue_detector = self.classifier_cls(BasicTissueMaskPredictor(model))
        tissue_detector.classify(slide)
        for patch in slide.patches:
            self.assertEqual(patch.features[TissueFeature.TISSUE_PERCENTAGE],
                             1)

    def test_mask(self):
        slide = Slide('data/test.tif', extraction_level=0)
        tissue_detector = self.classifier_cls(
            BasicTissueMaskPredictor(self.model_cls()))

        tissue_detector.classify(slide, include_mask_feature=True)
        for patch in slide.patches:
            if (patch.y == 0):
                self.assertEqual(
                    patch.features[TissueFeature.TISSUE_PERCENTAGE], 1)
                self.assertEqual(
                    patch.features[TissueFeature.TISSUE_MASK].all(), 1)

            else:
                self.assertEqual(
                    patch.features[TissueFeature.TISSUE_PERCENTAGE], 0)

                self.assertEqual(patch.features[TissueFeature.TISSUE_MASK],
                                 None)


class InterpolatedTissueClassifierTest(TestTissueClassifierTest,
                                       unittest.TestCase):
    classifier_cls = InterpolatedTissueClassifier
    predictor_cls = BasicTissueMaskPredictor
    model_cls = GreenIsTissueModel


class BasicTissueClassifierTest(TestTissueClassifierTest, unittest.TestCase):
    classifier_cls = BasicTissueClassifier
    predictor_cls = BasicTissueMaskPredictor
    model_cls = GreenIsTissueModel


class EddlTissueClassifierTest(TestTissueClassifierTest, unittest.TestCase):
    classifier_cls = BasicTissueClassifier
    predictor_cls = BasicTissueMaskPredictor
    model_cls = EddlGreenIsTissueModel


class KarolinskaTest(unittest.TestCase):
    def test_true_value(self):
        size = (1024, 256)
        patch_size = (256, 256)
        mask_slide = Slide('data/karolinska-mask.tif', extraction_level=0)
        slide = DummySlide('slide', size, patch_size=patch_size)
        cl = KarolinskaTrueValueClassifier(mask_slide)
        slide_classified = cl.classify(slide)
        self.assertEqual(len(slide.patches), len(slide_classified.patches))
        for i, patch in enumerate(slide_classified.patches):
            feature = patch.features[KarolinskaFeature.CANCER_PERCENTAGE]
            if i < 1:
                self.assertEqual(feature, 1)
            else:
                self.assertEqual(feature, 0)


if __name__ == '__main__':
    unittest.main()
