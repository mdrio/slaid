import unittest

import numpy as np
import pytest

from slaid.classifiers import (
    BasicClassifier,
    FilteredPatchClassifier,
    FilteredPixelClassifier,
    PixelClassifier,
)
from slaid.commons import Mask
from slaid.commons.base import Filter, ImageInfo, Slide
from slaid.commons.dask import init_client
from slaid.commons.ecvl import BasicSlide as EcvlSlide
from slaid.commons.openslide import BasicSlide as OpenSlide
from slaid.models.eddl import TissueModel, TumorModel
from tests.commons import EddlGreenPatchModel, GreenModel


@pytest.mark.parametrize("image_info", [ImageInfo.create("bgr", "yx", "first")])
@pytest.mark.parametrize("level", [0])
@pytest.mark.parametrize("basic_slide_cls", [EcvlSlide, OpenSlide])
@pytest.mark.parametrize("classifier_cls", [PixelClassifier])
@pytest.mark.parametrize("model", [GreenModel()])
@pytest.mark.parametrize("tile_size", [1024])
@pytest.mark.parametrize("chunk_size", [None, 100])
def test_classify_slide(green_slide, classifier_cls, model, level, chunk_size):
    classifier = classifier_cls(model, "test", chunk_size=chunk_size)
    mask = classifier.classify(green_slide, level=level)

    assert mask.array.shape == green_slide.level_dimensions[level][::-1]
    green_zone = int(300 // green_slide.level_downsamples[level])
    assert (mask.array[:green_zone, :] == 100).all()
    assert (mask.array[green_zone:, :] == 0).all()


@pytest.mark.parametrize("image_info", [ImageInfo.create("rgb", "yx", "last")])
@pytest.mark.parametrize("level", [0])
@pytest.mark.parametrize("basic_slide_cls", [OpenSlide, EcvlSlide])
@pytest.mark.parametrize("classifier_cls", [FilteredPixelClassifier])
@pytest.mark.parametrize("model", [GreenModel()])
@pytest.mark.parametrize("tile_size", [None])
def test_classify_with_filter(green_slide, classifier_cls, level, model):
    filter_level = 2
    filter_downsample = green_slide.level_downsamples[filter_level]
    filter_array = np.zeros(green_slide.level_dimensions[filter_level][::-1])
    tile_size = green_slide[0].size[0] // filter_array.shape[0]
    green_slide._store.tile_size = tile_size
    ones_row = 10
    ones_col = 10
    filter_array[:ones_row, :ones_col] = 1
    filter_mask = Mask(
        filter_array, filter_level, filter_downsample, green_slide.level_dimensions
    )
    classifier = classifier_cls(model, "test", filter_mask >= 1)
    mask = classifier.classify(green_slide, level=level)

    ones_row = round(
        ones_row * filter_downsample // green_slide.level_downsamples[level]
    )
    assert mask.array.shape == green_slide.level_dimensions[level][::-1]
    assert (mask.array[:ones_row, :ones_col] == 100).all()
    assert (mask.array[ones_row:, :ones_col] == 0).all()


@pytest.mark.parametrize("image_info", [ImageInfo.create("rgb", "yx", "last")])
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("basic_slide_cls", [EcvlSlide, OpenSlide])
@pytest.mark.parametrize("classifier_cls", [FilteredPixelClassifier])
@pytest.mark.parametrize("model", [GreenModel()])
@pytest.mark.parametrize("tile_size", [None])
def test_classify_with_zeros_as_filter(green_slide, classifier_cls, level, model):
    filter_level = 2
    filter_downsample = green_slide.level_downsamples[filter_level]
    filter_array = np.zeros(green_slide.level_dimensions[filter_level][::-1])
    filter_mask = Mask(
        filter_array, filter_level, filter_downsample, green_slide.level_dimensions
    )

    classifier = classifier_cls(model, "test", filter_mask >= 1)
    mask = classifier.classify(green_slide, level=level)

    assert mask.array.shape == green_slide.level_dimensions[level][::-1]
    assert (mask.array[:, :] == 0).all()


@pytest.mark.parametrize("image_info", [ImageInfo.create("bgr", "yx", "first")])
@pytest.mark.parametrize("level", [0])
@pytest.mark.parametrize("basic_slide_cls", [EcvlSlide, OpenSlide])
@pytest.mark.parametrize("classifier_cls", [BasicClassifier])
@pytest.mark.parametrize("model", [EddlGreenPatchModel((50, 50))])
@pytest.mark.parametrize("tile_size", [50])
def test_classify_slide_by_patches(green_slide, classifier, level):
    mask = classifier.classify(green_slide, level=level)
    dims = green_slide.level_dimensions[level][::-1]
    assert mask.array.shape == (
        dims[0] // classifier.model.patch_size[0],
        dims[1] // classifier.model.patch_size[1],
    )
    green_zone = int(
        300 // green_slide.level_downsamples[level] // classifier.model.patch_size[0]
    )
    assert (mask.array[:green_zone, :] == 100).all()
    assert (mask.array[green_zone:, :] == 0).all()
    assert mask.tile_size == 50


@pytest.mark.parametrize("image_info", [ImageInfo.create("bgr", "yx", "first")])
@pytest.mark.parametrize("level", [0])
@pytest.mark.parametrize("basic_slide_cls", [EcvlSlide, OpenSlide])
@pytest.mark.parametrize("classifier_cls", [FilteredPatchClassifier])
@pytest.mark.parametrize("model", [EddlGreenPatchModel((50, 50))])
@pytest.mark.parametrize("tile_size", [50])
def test_classify_slide_by_patches_with_filter_all_zeros(
    classifier_cls, green_slide, model, level
):
    filter_array = np.zeros(
        (
            green_slide.level_dimensions[level][1] // model.patch_size[0],
            green_slide.level_dimensions[level][0] // model.patch_size[1],
        ),
        dtype="bool",
    )

    classifier = classifier_cls(model, "test", Filter(None, filter_array))
    mask = classifier.classify(green_slide, level=level)
    dims = green_slide.level_dimensions[level][::-1]
    assert mask.array.shape == (
        dims[0] // model.patch_size[0],
        dims[1] // model.patch_size[1],
    )
    assert (mask.array == 0).all()


@pytest.mark.parametrize("image_info", [ImageInfo.create("bgr", "yx", "first")])
@pytest.mark.parametrize("level", [0])
@pytest.mark.parametrize("basic_slide_cls", [EcvlSlide, OpenSlide])
@pytest.mark.parametrize("classifier_cls", [FilteredPatchClassifier])
@pytest.mark.parametrize("model", [EddlGreenPatchModel((50, 50))])
@pytest.mark.parametrize("tile_size", [50])
def test_classify_slide_by_patches_with_filter(
    classifier_cls, green_slide, level, model
):
    filter_array = np.zeros(
        (
            green_slide.level_dimensions[level][1] // model.patch_size[0],
            green_slide.level_dimensions[level][0] // model.patch_size[1],
        ),
        dtype="bool",
    )
    filter_array[1, 1] = True
    classifier = classifier_cls(model, "test", Filter(None, filter_array))
    mask = classifier.classify(green_slide, level=level)

    dims = green_slide.level_dimensions[level][::-1]
    assert mask.array.shape == (
        dims[0] // classifier.model.patch_size[0],
        dims[1] // classifier.model.patch_size[1],
    )
    assert (mask.array[0, :] == 0).all()
    assert (mask.array[1, 0] == 0).all()
    assert (mask.array[1, 1] == 100).all()
    assert (mask.array[1, 2:] == 0).all()
    assert (mask.array[2:, :] == 0).all()


@pytest.mark.parametrize("slide_path", ["tests/data/patch.tif"])
@pytest.mark.parametrize("basic_slide_cls", [EcvlSlide, OpenSlide])
@pytest.mark.parametrize("classifier_cls", [BasicClassifier])
@pytest.mark.parametrize("slide_cls", [Slide])
@pytest.mark.parametrize("image_info", [ImageInfo.create("bgr", "yx", "first")])
@pytest.mark.parametrize(
    "model",
    [TumorModel("slaid/resources/models/promort_vgg16_weights_ep_9_vacc_0.85.bin")],
)
def test_classifies_tumor(slide, classifier, patch_tissue_mask):
    mask = classifier.classify(slide, level=0, round_to_0_100=False)
    assert round(float(mask.array[0]), 4) == round(0.11082522, 4)

    classifier.set_filter(Filter(None, np.ones((1, 1), dtype="bool")))
    mask = classifier.classify(slide, level=0, round_to_0_100=False)

    print(mask.array.shape, type(mask.array))
    assert round(float(mask.array[0]), 4) == round(1 - 0.11082522, 4)


@pytest.mark.parametrize("slide_path", ["tests/data/patch.tif"])
@pytest.mark.parametrize("basic_slide_cls", [OpenSlide, EcvlSlide])
@pytest.mark.parametrize("slide_cls", [Slide])
@pytest.mark.parametrize("image_info", [ImageInfo.create("rgb", "yx", "last")])
def test_classifies_tissue(slide, patch_tissue_mask):
    model = TissueModel(
        "slaid/resources/models/tissue_model-extract_tissue_eddl_1.1.bin"
    )
    classifier = PixelClassifier(model, "tissue")
    mask = classifier.classify(slide, level=0, round_to_0_100=False)
    assert (mask.array == patch_tissue_mask).all()

    downsample = 8
    slide._store.tile_size = 8
    filter_array = np.zeros(
        (
            patch_tissue_mask.shape[0] // downsample,
            patch_tissue_mask.shape[1] // downsample,
        ),
        dtype="bool",
    )
    filter_array[:2, :2] = 1

    filter_classifier = FilteredPixelClassifier(
        model, "tissue", Filter(None, filter_array)
    )
    mask = filter_classifier.classify(
        slide,
        level=0,
        round_to_0_100=False,
    )

    assert (mask.array[:16, :16] == patch_tissue_mask[:16, :16]).all()


if __name__ == "__main__":
    init_client()
    unittest.main()
