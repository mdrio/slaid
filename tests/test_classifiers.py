import unittest

import numpy as np
import pytest

from slaid.classifiers import BasicClassifier, PatchClassifier
from slaid.commons import Mask
from slaid.commons.base import Filter, ImageInfo, Slide
from slaid.commons.dask import init_client
from slaid.commons.ecvl import BasicSlide as EcvlSlide
from slaid.commons.openslide import Slide as OpenSlide
from slaid.models.eddl import TissueModel, TumorModel
from tests.commons import EddlGreenPatchModel, GreenModel


@pytest.mark.parametrize("image_info", [ImageInfo.create("bgr", "yx", "first")])
@pytest.mark.parametrize("level", [0])
@pytest.mark.parametrize("basic_slide_cls", [EcvlSlide, OpenSlide])
@pytest.mark.parametrize("classifier_cls", [BasicClassifier])
@pytest.mark.parametrize("model", [GreenModel()])
@pytest.mark.parametrize("tile_size", [1024])
def test_classify_slide(green_slide, classifier, level):
    mask = classifier.classify(green_slide, level=level)

    assert mask.array.shape == green_slide.level_dimensions[level][::-1]
    green_zone = int(300 // green_slide.level_downsamples[level])
    assert (mask.array[:green_zone, :] == 100).all()
    assert (mask.array[green_zone:, :] == 0).all()


@pytest.mark.parametrize("image_info", [ImageInfo.create("rgb", "yx", "last")])
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("max_MB_prediction", [None])
@pytest.mark.parametrize("basic_slide_cls", [EcvlSlide, OpenSlide])
@pytest.mark.parametrize("classifier_cls", [BasicClassifier])
@pytest.mark.parametrize("model", [GreenModel()])
@pytest.mark.parametrize("tile_size", [1024])
def test_classify_with_filter(green_slide, classifier, level, max_MB_prediction):
    filter_level = 2
    filter_downsample = green_slide.level_downsamples[filter_level]
    filter_array = np.zeros(green_slide.level_dimensions[filter_level][::-1])
    ones_row = 50
    filter_array[:ones_row, :] = 1
    filter_mask = Mask(
        filter_array, filter_level, filter_downsample, green_slide.level_dimensions
    )

    mask = classifier.classify(green_slide, level=level, filter_=filter_mask >= 1)

    ones_row = round(
        ones_row * filter_downsample // green_slide.level_downsamples[level]
    )
    assert mask.array.shape == green_slide.level_dimensions[level][::-1]
    assert (mask.array[:ones_row, :] == 100).all()
    assert (mask.array[ones_row:, :] == 0).all()


@pytest.mark.parametrize("image_info", [ImageInfo.create("rgb", "yx", "last")])
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("max_MB_prediction", [None])
@pytest.mark.parametrize("basic_slide_cls", [EcvlSlide, OpenSlide])
@pytest.mark.parametrize("classifier_cls", [BasicClassifier])
@pytest.mark.parametrize("model", [GreenModel()])
@pytest.mark.parametrize("tile_size", [1024])
def test_classify_with_zeros_as_filter(
    green_slide, classifier, level, max_MB_prediction
):
    filter_level = 2
    filter_downsample = green_slide.level_downsamples[filter_level]
    filter_array = np.zeros(green_slide.level_dimensions[filter_level][::-1])
    filter_mask = Mask(
        filter_array, filter_level, filter_downsample, green_slide.level_dimensions
    )

    mask = classifier.classify(green_slide, level=level, filter_=filter_mask >= 1)

    assert mask.array.shape == green_slide.level_dimensions[level][::-1]
    assert (mask.array[:, :] == 0).all()


@pytest.mark.parametrize("image_info", [ImageInfo.create("bgr", "yx", "first")])
@pytest.mark.parametrize("level", [0])
@pytest.mark.parametrize("basic_slide_cls", [EcvlSlide, OpenSlide])
@pytest.mark.parametrize("classifier_cls", [BasicClassifier, PatchClassifier])
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
@pytest.mark.parametrize("classifier_cls", [BasicClassifier, PatchClassifier])
@pytest.mark.parametrize("model", [EddlGreenPatchModel((50, 50))])
@pytest.mark.parametrize("tile_size", [50])
def test_classify_slide_by_patches_with_filter_all_zeros(
    green_slide, classifier, level
):
    filter_array = np.zeros(
        (
            green_slide.level_dimensions[level][1] // classifier.model.patch_size[0],
            green_slide.level_dimensions[level][0] // classifier.model.patch_size[1],
        ),
        dtype="bool",
    )
    mask = classifier.classify(
        green_slide, level=level, filter_=Filter(None, filter_array)
    )
    dims = green_slide.level_dimensions[level][::-1]
    assert mask.array.shape == (
        dims[0] // classifier.model.patch_size[0],
        dims[1] // classifier.model.patch_size[1],
    )
    assert (mask.array == 0).all()


@pytest.mark.parametrize("image_info", [ImageInfo.create("bgr", "yx", "first")])
@pytest.mark.parametrize("level", [0])
@pytest.mark.parametrize("basic_slide_cls", [EcvlSlide, OpenSlide])
@pytest.mark.parametrize("classifier_cls", [BasicClassifier, PatchClassifier])
@pytest.mark.parametrize("model", [EddlGreenPatchModel((50, 50))])
@pytest.mark.parametrize("tile_size", [50])
def test_classify_slide_by_patches_with_filter(green_slide, classifier, level):
    filter_array = np.zeros(
        (
            green_slide.level_dimensions[level][1] // classifier.model.patch_size[0],
            green_slide.level_dimensions[level][0] // classifier.model.patch_size[1],
        ),
        dtype="bool",
    )
    filter_array[1, 1] = True
    mask = classifier.classify(
        green_slide, level=level, filter_=Filter(None, filter_array)
    )
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
@pytest.mark.parametrize("classifier_cls", [BasicClassifier, PatchClassifier])
@pytest.mark.parametrize("slide_cls", [Slide])
@pytest.mark.parametrize("image_info", [ImageInfo.create("bgr", "yx", "first")])
@pytest.mark.parametrize(
    "model", ["slaid/resources/models/promort_vgg16_weights_ep_9_vacc_0.85.bin"]
)
def test_classifies_tumor(slide, classifier, patch_tissue_mask):
    mask = classifier.classify(slide, level=0, round_to_0_100=False)
    assert round(float(mask.array[0]), 4) == round(1 - 0.11082522, 4)

    mask = classifier.classify(
        slide,
        level=0,
        round_to_0_100=False,
        filter_=Filter(None, np.ones((1, 1), dtype="bool")),
    )
    print(mask.array.shape, type(mask.array))
    assert round(float(mask.array[0]), 4) == round(1 - 0.11082522, 4)


@pytest.mark.parametrize("slide_path", ["tests/data/patch.tif"])
@pytest.mark.parametrize("basic_slide_cls", [EcvlSlide, OpenSlide])
@pytest.mark.parametrize("slide_cls", [Slide])
@pytest.mark.parametrize("image_info", [ImageInfo.create("rgb", "yx", "last")])
def test_classifies_tissue(slide, patch_tissue_mask):
    model = TissueModel(
        "slaid/resources/models/tissue_model-extract_tissue_eddl_1.1.bin"
    )
    classifier = BasicClassifier(model, "tissue")
    mask = classifier.classify(slide, level=0, round_to_0_100=False)
    assert (mask.array == patch_tissue_mask).all()

    mask = classifier.classify(
        slide,
        level=0,
        round_to_0_100=False,
        filter_=Filter(None, np.ones(patch_tissue_mask.shape, dtype="bool")),
    )
    assert (mask.array == patch_tissue_mask).all()


if __name__ == "__main__":
    init_client()
    unittest.main()
