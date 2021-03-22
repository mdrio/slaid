import unittest

import numpy as np
import pytest
import tiledb

from slaid.commons import Mask
from slaid.commons.base import ImageInfo
from slaid.commons.dask import Mask as DaskMask
from slaid.commons.ecvl import BasicSlide as EcvlSlide

IMAGE = 'tests/data/test.tif'


@pytest.mark.parametrize('image_info', [
    ImageInfo('bgr', 'yx', 'first'),
    ImageInfo('rgb', 'yx', 'first'),
    ImageInfo('rgb', 'yx', 'last'),
    ImageInfo('bgr', 'yx', 'last')
])
@pytest.mark.parametrize('slide_cls', [EcvlSlide])
def test_slide_level(slide):
    for i in range(slide.level_count):
        array = slide[i]
        assert array.size == slide.level_dimensions[i][::-1]


@pytest.mark.parametrize('image_info', [
    ImageInfo('bgr', 'yx', 'first'),
    ImageInfo('rgb', 'yx', 'first'),
    ImageInfo('rgb', 'yx', 'last'),
    ImageInfo('bgr', 'yx', 'last')
])
@pytest.mark.parametrize('slide_cls', [EcvlSlide])
def test_slice_slide(slide):
    for i in range(slide.level_count):
        array = slide[i]
        expected_shape = (
            3, 10,
            20) if slide.image_info.channel == ImageInfo.CHANNEL.FIRST else (
                10, 20, 3)
        assert array[:10, :20]._array.shape == expected_shape


@pytest.mark.parametrize('image_info', [ImageInfo('bgr', 'yx', 'first')])
@pytest.mark.parametrize('slide_cls', [EcvlSlide])
def test_reads_region(slide):
    image = slide.read_region((0, 0), 0, (256, 256))
    image.dimensions == (256, 256)


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


def test_slide_array_reshape(slide_array):
    size = slide_array.size
    slide_array = slide_array.reshape((size[0] * size[1], 1))
    assert slide_array.size == (size[0] * size[1], 1)


if __name__ == '__main__':
    unittest.main()
