import unittest
from datetime import datetime as dt

import numpy as np
import pytest

from slaid.classifiers import BasicClassifier
from slaid.classifiers.dask import Classifier as DaskClassifier
from slaid.commons import Mask
from slaid.commons.base import Filter, ImageInfo, Slide
from slaid.commons.dask import DaskSlide, init_client
from slaid.commons.ecvl import BasicSlide as EcvlSlide
from slaid.models.eddl import TissueModel, TumorModel


@pytest.mark.parametrize('image_info', [ImageInfo('bgr', 'yx', 'first')])
@pytest.mark.parametrize('level', [0, 1])
@pytest.mark.parametrize('backend', ['basic', 'dask'])
@pytest.mark.parametrize('max_MB_prediction', [None])
def test_classify_slide(green_slide_and_classifier, level, max_MB_prediction):
    green_slide, green_classifier = green_slide_and_classifier
    mask = green_classifier.classify(green_slide,
                                     level=level,
                                     max_MB_prediction=max_MB_prediction)
    assert mask.array.shape == green_slide.level_dimensions[level][::-1]
    green_zone = int(300 // green_slide.level_downsamples[level])
    print(mask.array)
    assert (mask.array[:green_zone, :] == 100).all()
    assert (mask.array[green_zone:, :] == 0).all()


@pytest.mark.parametrize('image_info', [ImageInfo('bgr', 'yx', 'first')])
@pytest.mark.parametrize('level', [0, 1])
@pytest.mark.parametrize('backend', ['basic', 'dask'])
@pytest.mark.parametrize('max_MB_prediction', [None])
def test_classify_with_filter(green_slide_and_classifier, level,
                              max_MB_prediction):
    green_slide, green_classifier = green_slide_and_classifier
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


@pytest.mark.parametrize('image_info', [ImageInfo('bgr', 'yx', 'first')])
@pytest.mark.parametrize('level', [0])
#  @pytest.mark.parametrize('backend', ['basic', 'dask'])
@pytest.mark.parametrize('backend', ['dask'])
@pytest.mark.parametrize('max_MB_prediction', [None])
def test_classify_slide_by_patches(green_slide_and_patch_classifier, level,
                                   max_MB_prediction):
    green_slide, green_classifier = green_slide_and_patch_classifier
    mask = green_classifier.classify(green_slide,
                                     level=level,
                                     max_MB_prediction=max_MB_prediction)
    dims = green_slide.level_dimensions[level][::-1]
    assert mask.array.shape == (dims[0] //
                                green_classifier.model.patch_size[0],
                                dims[1] //
                                green_classifier.model.patch_size[1])
    green_zone = int(300 // green_slide.level_downsamples[level] //
                     green_classifier.model.patch_size[0])
    assert (mask.array[:green_zone, :] == 100).all()
    assert (mask.array[green_zone:, :] == 0).all()


@pytest.mark.parametrize('image_info', [ImageInfo('bgr', 'yx', 'first')])
@pytest.mark.parametrize('level', [0])
#  @pytest.mark.parametrize('backend', ['basic', 'dask'])
@pytest.mark.parametrize('backend', ['dask'])
@pytest.mark.parametrize('max_MB_prediction', [None])
def test_classify_slide_by_patches_with_filter_all_zeros(
        green_slide_and_patch_classifier, level, max_MB_prediction):
    green_slide, green_classifier = green_slide_and_patch_classifier
    filter_array = np.zeros((green_slide.level_dimensions[level][1] //
                             green_classifier.model.patch_size[0],
                             green_slide.level_dimensions[level][0] //
                             green_classifier.model.patch_size[1]),
                            dtype='bool')
    mask = green_classifier.classify(green_slide,
                                     level=level,
                                     max_MB_prediction=max_MB_prediction,
                                     filter_=Filter(None, filter_array))
    dims = green_slide.level_dimensions[level][::-1]
    assert mask.array.shape == (dims[0] //
                                green_classifier.model.patch_size[0],
                                dims[1] //
                                green_classifier.model.patch_size[1])
    assert (mask.array == 0).all()


@pytest.mark.parametrize('image_info', [ImageInfo('bgr', 'yx', 'first')])
@pytest.mark.parametrize('level', [0])
#  @pytest.mark.parametrize('backend', ['basic', 'dask'])
@pytest.mark.parametrize('backend', ['dask'])
@pytest.mark.parametrize('max_MB_prediction', [None])
def test_classify_slide_by_patches_with_filter(
        green_slide_and_patch_classifier, level, max_MB_prediction):
    green_slide, green_classifier = green_slide_and_patch_classifier
    filter_array = np.zeros((green_slide.level_dimensions[level][1] //
                             green_classifier.model.patch_size[0],
                             green_slide.level_dimensions[level][0] //
                             green_classifier.model.patch_size[1]),
                            dtype='bool')
    filter_array[1, 1] = True
    mask = green_classifier.classify(green_slide,
                                     level=level,
                                     max_MB_prediction=max_MB_prediction,
                                     filter_=Filter(None, filter_array))
    dims = green_slide.level_dimensions[level][::-1]
    assert mask.array.shape == (dims[0] //
                                green_classifier.model.patch_size[0],
                                dims[1] //
                                green_classifier.model.patch_size[1])
    assert (mask.array[0, :] == 0).all()
    assert (mask.array[1, 0] == 0).all()
    assert (mask.array[1, 1] == 100).all()
    assert (mask.array[1, 2:] == 0).all()
    assert (mask.array[2:, :] == 0).all()


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

if __name__ == '__main__':
    init_client()
    unittest.main()
