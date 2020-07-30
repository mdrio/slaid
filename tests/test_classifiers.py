import unittest
from slaid.commons import Slide
from slaid.classifiers import BasicTissueMaskPredictor,\
    TissueClassifier, TissueFeature, \
    KarolinskaTrueValueClassifier, KarolinskaFeature
import numpy as np
from test_commons import DummySlide


class GreenIsTissueModel:
    def predict(self, array: np.array) -> np.array:
        return array[:, 1] / 255


class DummyModel:
    def __init__(self, func):
        self.func = func

    def predict(self, array):
        return self.func(array.shape[0])


class TestTissueDetector(unittest.TestCase):
    def test_detector_no_tissue(self):
        patch_size = (10, 10)
        slide = DummySlide('slide', (100, 100), patch_size=patch_size)
        model = DummyModel(np.zeros)
        tissue_detector = TissueClassifier(BasicTissueMaskPredictor(model))

        tissue_detector.classify(slide)
        for patch in slide.patches:
            self.assertEqual(patch.features[TissueFeature.TISSUE_PERCENTAGE],
                             0)

    def test_detector_all_tissue(self):
        patch_size = (10, 10)
        slide = DummySlide('slide', (100, 100), patch_size=patch_size)
        model = DummyModel(np.ones)
        tissue_detector = TissueClassifier(BasicTissueMaskPredictor(model))
        tissue_detector.classify(slide)
        for patch in slide.patches:
            self.assertEqual(patch.features[TissueFeature.TISSUE_PERCENTAGE],
                             1)

    def test_mask(self):
        slide = Slide('data/input.tiff', extraction_level=0)
        model = GreenIsTissueModel()
        tissue_detector = TissueClassifier(BasicTissueMaskPredictor(model))

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


class KarolinskaTest(unittest.TestCase):
    def test_true_value(self):
        size = (200, 100)
        patch_size = (10, 10)
        #  size = (23904, 28664)
        #  patch_size = (256, 256)
        data = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        data[0:10, 0:50] = [2, 0, 0]

        mask_slide = DummySlide('mask', size, data=data)
        slide = DummySlide('slide', size, patch_size=patch_size)
        cl = KarolinskaTrueValueClassifier(mask_slide)
        slide_classified = cl.classify(slide)
        self.assertEqual(len(slide.patches), len(slide_classified.patches))
        for i, patch in enumerate(slide_classified.patches):
            feature = patch.features[KarolinskaFeature.CANCER_PERCENTAGE]
            if i <= 4:
                self.assertEqual(feature, 1)
            else:
                self.assertEqual(feature, 0)


if __name__ == '__main__':
    unittest.main()
