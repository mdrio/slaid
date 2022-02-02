import unittest

import numpy as np
import pytest
import tiledb

from slaid.commons import Mask
from slaid.commons.base import ImageInfo, Slide, SlideArray
from slaid.commons.dask import DaskSlideArray
from slaid.commons.dask import Mask as DaskMask
from slaid.commons.dask import Slide as DaskSlide
from slaid.commons.ecvl import BasicSlide as EcvlSlide
from slaid.commons.openslide import BasicSlide as OpenSlide

IMAGE = 'tests/data/test.tif'


@pytest.mark.parametrize('image_info', [
    ImageInfo.create('bgr', 'yx', 'first'),
    ImageInfo.create('rgb', 'yx', 'first'),
    ImageInfo.create('rgb', 'yx', 'last'),
    ImageInfo.create('bgr', 'yx', 'last')
])
@pytest.mark.parametrize('basic_slide_cls', [EcvlSlide, OpenSlide])
@pytest.mark.parametrize('slide_cls', [Slide, DaskSlide])
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
@pytest.mark.parametrize('basic_slide_cls', [EcvlSlide, OpenSlide])
@pytest.mark.parametrize('slide_cls', [Slide, DaskSlide])
def test_slice_slide(slide, image_info):
    for i in range(slide.level_count):
        array = slide[i].convert(image_info)
        expected_shape = (
            3, 10,
            20) if slide.image_info.channel == ImageInfo.CHANNEL.FIRST else (
                10, 20, 3)
        assert array[:10, :20]._array.shape == expected_shape


@pytest.mark.parametrize('image_info',
                         [ImageInfo.create('bgr', 'yx', 'first')])
@pytest.mark.parametrize('basic_slide_cls', [EcvlSlide])
@pytest.mark.parametrize('slide_cls', [Slide, DaskSlide])
def test_read_region(slide):
    image = slide.read_region((0, 0), 0, (256, 256))
    assert image.dimensions == (256, 256)
    array = image.to_array()
    assert array.shape == (
        3,
        256,
        256,
    )


@pytest.mark.parametrize('image_info',
                         [ImageInfo.create('bgr', 'yx', 'first')])
@pytest.mark.parametrize('basic_slide_cls', [EcvlSlide, OpenSlide])
@pytest.mark.parametrize('slide_cls', [Slide])
def test_slice_read(slide):
    image = slide.read_region((0, 0), 0, slide.dimensions)
    slide_array = slide[0][:, :]
    image_array = image.to_array()
    assert (slide_array.array == image_array).all()


class TestMask:
    cls = Mask

    @pytest.mark.skip(reason="update how mask are loaded/dumped")
    def test_dumps_to_tiledb(self, array, tmp_path):
        mask = self.cls(array, 1, 1, 0.8, False)
        mask.to_tiledb(str(tmp_path))
        with tiledb.open(str(tmp_path), 'r') as array:
            assert (array == np.array(mask.array)).all()
            assert array.meta['extraction_level'] == mask.extraction_level
            assert array.meta['level_downsample'] == mask.level_downsample
            assert array.meta['threshold'] == mask.threshold
            assert 'model' not in array.meta.keys()

    @pytest.mark.skip(reason="update how mask are loaded/dumped")
    def test_creates_from_tiledb(self, tiledb_path):
        mask = self.cls.from_tiledb(tiledb_path)
        with tiledb.open(tiledb_path, 'r') as array:
            assert (mask.array[:] == array[:]).all()


class TestDaskMask(TestMask):
    cls = DaskMask

    @pytest.mark.skip(reason="update how mask are loaded/dumped")
    def test_dumps_to_tiledb(self, dask_array, tmp_path):
        super().test_dumps_to_tiledb(dask_array, tmp_path)


@pytest.mark.parametrize('cls', [SlideArray, DaskSlideArray])
def test_slide_array_reshape(slide_array):
    size = slide_array.size
    slide_array = slide_array.reshape((size[0] * size[1], 1))
    assert slide_array.size == (size[0] * size[1], 1)


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
