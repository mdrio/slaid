import unittest
from datetime import datetime as dt

import numpy as np
import pytest

from slaid.classifiers import BasicClassifier
from slaid.classifiers.dask import Classifier as DaskClassifier
from slaid.commons import Mask
from slaid.commons.base import Filter, ImageInfo, Slide
from slaid.commons.dask import Slide as DaskSlide
from slaid.commons.dask import init_client
from slaid.commons.ecvl import BasicSlide as EcvlSlide
from slaid.models.eddl import TissueModel, TumorModel


@pytest.mark.parametrize('image_info', [ImageInfo('bgr', 'yx', 'first')])
@pytest.mark.parametrize('level', [0, 1])
@pytest.mark.parametrize('backend', ['basic', 'dask'])
def test_classify_slide(green_slide_and_classifier, level):
    green_slide, green_classifier = green_slide_and_classifier
    mask = green_classifier.classify(green_slide, level=level)

    assert mask.array.shape == green_slide.level_dimensions[level][::-1]
    green_zone = int(300 // green_slide.level_downsamples[level])
    assert (mask.array[:green_zone, :] == 100).all()
    assert (mask.array[green_zone:, :] == 0).all()


@pytest.mark.parametrize('image_info', [ImageInfo('rgb', 'yx', 'last')])
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
    filter_mask = Mask(filter_array, filter_level, filter_downsample,
                       green_slide.level_dimensions)

    mask = green_classifier.classify(green_slide,
                                     level=level,
                                     filter_=filter_mask >= 1)

    ones_row = round(ones_row * filter_downsample //
                     green_slide.level_downsamples[level])
    assert mask.array.shape == green_slide.level_dimensions[level][::-1]
    assert (mask.array[:ones_row, :] == 100).all()
    assert (mask.array[ones_row:, :] == 0).all()


@pytest.mark.parametrize('image_info', [ImageInfo('rgb', 'yx', 'last')])
@pytest.mark.parametrize('level', [0, 1])
@pytest.mark.parametrize('backend', ['basic', 'dask'])
@pytest.mark.parametrize('max_MB_prediction', [None])
def test_classify_with_zeros_as_filter(green_slide_and_classifier, level,
                                       max_MB_prediction):
    green_slide, green_classifier = green_slide_and_classifier
    filter_level = 2
    filter_downsample = green_slide.level_downsamples[filter_level]
    filter_array = np.zeros(green_slide.level_dimensions[filter_level][::-1])
    filter_mask = Mask(filter_array, filter_level, filter_downsample,
                       green_slide.level_dimensions)

    mask = green_classifier.classify(green_slide,
                                     level=level,
                                     filter_=filter_mask >= 1)

    assert mask.array.shape == green_slide.level_dimensions[level][::-1]
    assert (mask.array[:, :] == 0).all()


@pytest.mark.parametrize('image_info', [ImageInfo('bgr', 'yx', 'first')])
@pytest.mark.parametrize('level', [0])
@pytest.mark.parametrize('backend', ['basic', 'dask'])
def test_classify_slide_by_patches(green_slide_and_patch_classifier, level):
    green_slide, green_classifier = green_slide_and_patch_classifier
    mask = green_classifier.classify(green_slide, level=level)

    dims = green_slide.level_dimensions[level][::-1]
    assert mask.array.shape == (dims[0] //
                                green_classifier.model.patch_size[0],
                                dims[1] //
                                green_classifier.model.patch_size[1])
    green_zone = int(300 // green_slide.level_downsamples[level] //
                     green_classifier.model.patch_size[0])
    print(mask.array[green_zone:, :])
    assert (mask.array[:green_zone, :] == 100).all()
    assert (mask.array[green_zone:, :] == 0).all()


@pytest.mark.parametrize('image_info', [ImageInfo('bgr', 'yx', 'first')])
@pytest.mark.parametrize('level', [0])
@pytest.mark.parametrize('backend', ['basic', 'dask'])
def test_classify_slide_by_patches_with_filter_all_zeros(
        green_slide_and_patch_classifier, level):
    green_slide, green_classifier = green_slide_and_patch_classifier
    filter_array = np.zeros((green_slide.level_dimensions[level][1] //
                             green_classifier.model.patch_size[0],
                             green_slide.level_dimensions[level][0] //
                             green_classifier.model.patch_size[1]),
                            dtype='bool')
    mask = green_classifier.classify(green_slide,
                                     level=level,
                                     filter_=Filter(None, filter_array))
    dims = green_slide.level_dimensions[level][::-1]
    assert mask.array.shape == (dims[0] //
                                green_classifier.model.patch_size[0],
                                dims[1] //
                                green_classifier.model.patch_size[1])
    assert (mask.array == 0).all()


@pytest.mark.parametrize('image_info', [ImageInfo('bgr', 'yx', 'first')])
@pytest.mark.parametrize('level', [0])
@pytest.mark.parametrize('backend', ['basic', 'dask'])
def test_classify_slide_by_patches_with_filter(
        green_slide_and_patch_classifier, level):
    green_slide, green_classifier = green_slide_and_patch_classifier
    filter_array = np.zeros((green_slide.level_dimensions[level][1] //
                             green_classifier.model.patch_size[0],
                             green_slide.level_dimensions[level][0] //
                             green_classifier.model.patch_size[1]),
                            dtype='bool')
    filter_array[1, 1] = True
    mask = green_classifier.classify(green_slide,
                                     level=level,
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


@pytest.mark.parametrize('slide_path', ['tests/data/patch.tif'])
@pytest.mark.parametrize('basic_slide_cls', [EcvlSlide])
@pytest.mark.parametrize('slide_cls', [DaskSlide])
@pytest.mark.parametrize('image_info', [ImageInfo('bgr', 'yx', 'first')])
def test_classifies_tumor(slide, patch_tissue_mask):
    model = TumorModel(
        'slaid/resources/models/promort_vgg16_weights_ep_9_vacc_0.85.bin')
    classifier = DaskClassifier(model, 'tumor')
    mask = classifier.classify(slide, level=0, round_to_0_100=False)
    assert round(float(mask.array[0]), 4) == round(0.11082522, 4)

    mask = classifier.classify(slide,
                               level=0,
                               round_to_0_100=False,
                               filter_=Filter(None,
                                              np.ones((1, 1), dtype='bool')))
    print(mask.array.shape, type(mask.array))
    assert round(float(mask.array[0]), 4) == round(0.11082522, 4)


@pytest.mark.parametrize('slide_path', ['tests/data/patch.tif'])
@pytest.mark.parametrize('basic_slide_cls', [EcvlSlide])
@pytest.mark.parametrize('slide_cls', [DaskSlide])
@pytest.mark.parametrize('image_info', [ImageInfo('rgb', 'yx', 'last')])
def test_classifies_tissue(slide, patch_tissue_mask):
    model = TissueModel(
        'slaid/resources/models/tissue_model-extract_tissue_eddl_1.1.bin')
    classifier = DaskClassifier(model, 'tissue')
    mask = classifier.classify(slide, level=0, round_to_0_100=False)
    assert (mask.array == patch_tissue_mask).all()

    mask = classifier.classify(slide,
                               level=0,
                               round_to_0_100=False,
                               filter_=Filter(
                                   None,
                                   np.ones(patch_tissue_mask.shape,
                                           dtype='bool')))
    assert (mask.array == patch_tissue_mask).all()


if __name__ == '__main__':
    init_client()
    unittest.main()
