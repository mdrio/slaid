import unittest
from datetime import datetime as dt

import numpy as np
import pytest

from slaid.classifiers import BasicClassifier
from slaid.classifiers.dask import Classifier as DaskClassifier
from slaid.commons import Mask
from slaid.commons.base import ImageInfo, Slide
from slaid.commons.dask import DaskSlide, init_client
from slaid.commons.ecvl import BasicSlide as EcvlSlide
from slaid.models.eddl import TissueModel, TumorModel


@pytest.mark.parametrize('image_info', [ImageInfo('bgr', 'yx', 'first')])
@pytest.mark.parametrize('basic_slide_cls', [EcvlSlide])
@pytest.mark.parametrize('slide_cls', [DaskSlide])
@pytest.mark.parametrize('classifier_cls', [DaskClassifier])
@pytest.mark.parametrize('level', [0, 1])
@pytest.mark.parametrize('max_MB_prediction', [None, 0.1])
def test_classify_slide(green_slide, green_classifier, level,
                        max_MB_prediction):
    mask = green_classifier.classify(green_slide,
                                     level=level,
                                     max_MB_prediction=max_MB_prediction)
    assert mask.array.shape == green_slide.level_dimensions[level][::-1]
    green_zone = int(300 // green_slide.level_downsamples[level])
    assert (mask.array[:green_zone, :] == 100).all()
    assert (mask.array[green_zone:, :] == 0).all()


@pytest.mark.parametrize('image_info', [ImageInfo('bgr', 'yx', 'first')])
@pytest.mark.parametrize('basic_slide_cls', [EcvlSlide])
@pytest.mark.parametrize('slide_cls', [Slide, DaskSlide])
@pytest.mark.parametrize('classifier_cls', [BasicClassifier, DaskClassifier])
@pytest.mark.parametrize('level', [0, 1])
@pytest.mark.parametrize('max_MB_prediction', [None, 0.01])
def test_classify_with_filter(green_slide, green_classifier, level,
                              max_MB_prediction):
    filter_level = 2
    filter_downsample = green_slide.level_downsamples[filter_level]
    filter_array = np.zeros(green_slide.level_dimensions[filter_level][::-1])
    ones_row = 50
    filter_array[:ones_row, :] = 1
    filter_mask = Mask(filter_array, filter_level, filter_downsample)

    mask = green_classifier.classify(green_slide,
                                     level=level,
                                     filter_=filter_mask >= 1,
                                     max_MB_prediction=max_MB_prediction)

    ones_row = round(ones_row * filter_downsample //
                     green_slide.level_downsamples[level])
    assert (mask.array[:ones_row, :] == 100).all()
    assert (mask.array[ones_row:, :] == 0).all()


#  class TestDaskClassifier(BaseTestClassifier, unittest.TestCase):
#      @classmethod
#      def setUpClass(cls):
#          #  init_client()
#          import dask
#          dask.config.set(scheduler='synchronous')
#
#      @staticmethod
#      def get_classifier(model, feature='tissue'):
#          return DaskClassifier(model, feature)
#
#      @staticmethod
#      def get_model():
#          return GreenModel()
#
#      def test_filter(self):
#          slide = load('tests/data/PH10023-1.thumb.tif')
#          model = load_model(
#              'slaid/resources/models/tissue_model-extract_tissue_eddl_1.1.bin')
#          classifier = self.get_classifier(model, 'low_level_tissue')
#          low_level_tissue = classifier.classify(slide, level=2)
#          level = 1
#          high_level_tissue = self.get_classifier(
#              model,
#              'high_level_tissue').classify(slide,
#                                            level=level,
#                                            filter_=low_level_tissue > 0.1)
#          self.assertEqual(high_level_tissue.array.shape[::-1],
#                           slide.level_dimensions[level])
#
#
#  class TestEddlClassifier(BasicClassifierTest):
#      @staticmethod
#      def get_model():
#          return EddlGreenModel()
#
#
#  class TestEddlPatchClassifier(unittest.TestCase):
#      @staticmethod
#      def get_model(patch_size):
#          return EddlGreenPatchModel(patch_size)
#
#      @staticmethod
#      def get_classifier(model, feature='tissue'):
#          return BasicClassifier(model, feature)
#
#      def test_classifies_by_patch_at_level_0(self, n_batch=1):
#          level = 0
#          slide = load('tests/data/test.tif')
#          patch_size = (100, 256)
#          classifier = self.get_classifier(self.get_model(patch_size))
#          mask = classifier.classify(slide, level=level, n_batch=n_batch)
#
#          self.assertEqual(
#              mask.array.shape,
#              tuple([
#                  slide.level_dimensions[level][::-1][i] // patch_size[i]
#                  for i in range(2)
#              ]),
#          )
#          print(mask.array)
#          self.assertEqual(mask.array[:3, :].all(), 1)
#          self.assertEqual(mask.array[3:, :].all(), 0)
#
#      def test_classifies_with_filter(self):
#          level = 0
#          patch_size = (100, 100)
#          slide = load('tests/data/test.tif')
#          mask_array = np.zeros(
#              np.array(slide.level_dimensions[level][::-1]) //
#              np.array(patch_size))
#          mask_array[0, :] = 1
#          filter_mask = Mask(mask_array, level, slide.level_downsamples[level],
#                             dt.now(), False)
#
#          slide.masks['tissue'] = filter_mask
#          classifier = self.get_classifier(self.get_model(patch_size), 'cancer')
#          mask = classifier.classify(slide, level=level, filter_=filter_mask > 0)
#
#          self.assertTrue((mask.array / 100 == filter_mask.array).all())
#
#      def test_classifies_with_no_filtered_patch(self):
#          level = 0
#          patch_size = (100, 100)
#          slide = load('tests/data/test.tif')
#          mask_array = np.zeros(
#              np.array(slide.level_dimensions[level][::-1]) //
#              np.array(patch_size))
#          filter_mask = Mask(mask_array, level, slide.level_downsamples[level],
#                             dt.now(), False)
#
#          slide.masks['tissue'] = filter_mask
#          classifier = self.get_classifier(self.get_model(patch_size), 'cancer')
#          mask = classifier.classify(slide, level=level, filter_=filter_mask > 0)
#
#          self.assertEqual(mask.array.shape, filter_mask.array.shape)
#          print(mask.array)
#          self.assertTrue(not np.count_nonzero(mask.array))
#


class TestFilter(unittest.TestCase):
    def test_filters_at_same_level(self):
        array = np.zeros((10, 10))
        indexes_ones = (0, 0)
        array[indexes_ones] = 1
        mask = Mask(array, 0, 1, dt.now(), False)
        filtered = mask > 0.5

        self.assertTrue((filtered.indices == indexes_ones).all())


def test_classifies_tumor(patch_path, slide_reader):
    slide = slide_reader(patch_path)
    model = TumorModel(
        'slaid/resources/models/promort_vgg16_weights_ep_9_vacc_0.85.bin')
    classifier = BasicClassifier(model, 'tumor')
    mask = classifier.classify(slide, level=0, round_to_0_100=False)
    assert round(float(mask.array[0]), 4) == round(0.11082522, 4)


def test_classifies_tissue(patch_path, slide_reader, patch_tissue_mask):
    slide = slide_reader(patch_path)
    model = TissueModel(
        'slaid/resources/models/tissue_model-extract_tissue_eddl_1.1.bin')
    classifier = BasicClassifier(model, 'tissue')
    mask = classifier.classify(slide, level=0, round_to_0_100=False)
    assert (mask.array == patch_tissue_mask).all()


if __name__ == '__main__':
    init_client()
    unittest.main()
