import unittest

import numpy as np
import pytest

from slaid.commons.base import ImageInfo, Slide
from slaid.commons.ecvl import BasicSlide as EcvlSlide
from slaid.commons.openslide import BasicSlide as OpenSlide

IMAGE = 'tests/data/test.tif'


@pytest.mark.parametrize('slide_cls,args', [(Slide, (EcvlSlide, )),
                                            (Slide, (OpenSlide, ))])
def test_slide_level(slide):
    for i in range(slide.level_count):
        array = slide[i]
        assert array.size == slide.level_dimensions[i][::-1]


@pytest.mark.parametrize('image_info', [
    ImageInfo.create('bgr', 'yx', 'first'),
    ImageInfo.create('rgb', 'yx', 'first'),
    ImageInfo.create('rgb', 'yx', 'last'),
    ImageInfo.create('bgr', 'yx', 'last')
])
@pytest.mark.parametrize('slide_cls,args', [(Slide, (EcvlSlide, )),
                                            (Slide, (OpenSlide, ))])
def test_slice_slide(slide, image_info):
    for i in range(slide.level_count):
        slide_array = slide[i]
        expected_shape = (
            3, 10,
            20) if image_info.channel == ImageInfo.Channel.FIRST else (10, 20,
                                                                       3)
        sliced_array = slide_array[:10, :20]
        sliced_array = sliced_array.convert(image_info)
        assert sliced_array.size == (10, 20)
        assert sliced_array.array.shape == expected_shape

        sliced_array = slide_array[1:10, 1:20]
        sliced_array = sliced_array.convert(image_info)
        assert sliced_array.size == (9, 19)

    image = slide.read_region((0, 0), 0, slide.dimensions)
    slide_array = slide[0][:, :]
    image_array = image.to_array()
    assert (slide_array.array == image_array).all()


def test_filter(mask):
    filter_ = mask >= 3
    assert (filter_[0, :] == 0).all()
    assert (filter_[1:, :] == 1).all()


def test_filter_rescale(mask):
    filter_ = mask >= 5
    print(filter_._array)
    filter_.rescale((9, 9))
    expected = [
        [False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, True, True, True],
        [False, False, False, False, False, False, True, True, True],
        [False, False, False, False, False, False, True, True, True],
        [True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True],
    ]
    print(filter_._array)
    assert (filter_[:, :] == np.array(expected)).all()


if __name__ == '__main__':
    unittest.main()
