import unittest

import numpy as np
import pytest
import tiledb

from slaid.commons import Mask, Slide
from slaid.commons.base import ImageInfo
from slaid.commons.dask import Mask as DaskMask
from slaid.commons.ecvl import Slide as EcvlSlide
from slaid.commons.ecvl import SlideArrayFactory
from slaid.commons.openslide import Slide as OpenSlide

IMAGE = 'tests/data/test.tif'


class BaseTestSlide:
    slide: Slide = None
    slide_cls = None

    def test_to_array_level_0(self):
        image_info = ImageInfo('bgr', 'yx', 'first')
        array = SlideArrayFactory(self.slide, image_info)[0]
        self.assertEqual(array.shape, self.slide.dimensions[::-1])

    def test_to_array_level_1(self):
        image_info = ImageInfo('bgr', 'yx', 'first')
        array = SlideArrayFactory(self.slide, image_info)[1]
        self.assertEqual(array.shape, self.slide.level_dimensions[1][::-1])

    def test_array_slice_channel_first(self):
        image_info = ImageInfo('bgr', 'yx', 'first')
        array = SlideArrayFactory(self.slide, image_info)[1]

        self.assertEqual(array[:20, :10]._array.shape, (3, 20, 10))

        self.assertEqual(array[0, 0]._array.shape, (3, ))

        import pudb
        pudb.set_trace()
        array = array[:20, :10]._array.compute()
        expected_array = np.array(self.slide.read_region((0, 0), 1, (10, 20)))
        print(array.shape, expected_array.shape)
        self.assertTrue(array == expected_array)

    def test_array_slice_channel_last(self):
        array = self.slide.to_array(1, ImageInfo('rgb', 'yx', 'last'))
        self.assertEqual(array[0, :].shape,
                         (self.slide.level_dimensions[1][0], 3))
        self.assertEqual(array[:20, :10].shape, (20, 10, 3))
        self.assertEqual(array[0, 0].shape, (3, ))

    def test_returns_dimensions(self):
        self.assertEqual(self.slide.dimensions, (512, 1024))

    def test_converts_to_array_xy_channel_first(self):
        region = self.slide.read_region((0, 0), 0, (10, 20))
        array = region.to_array(
            ImageInfo(ImageInfo.COLORTYPE.BGR, ImageInfo.COORD.XY,
                      ImageInfo.CHANNEL.FIRST))
        self.assertEqual(array.shape, (3, 10, 20))

    def test_converts_to_array_yx_channel_first(self):
        region = self.slide.read_region((0, 0), 0, (10, 20))
        array = region.to_array(
            ImageInfo(ImageInfo.COLORTYPE.BGR, ImageInfo.COORD.YX,
                      ImageInfo.CHANNEL.FIRST))
        self.assertEqual(array.shape, (3, 20, 10))

    def test_converts_to_array_xy_channel_last(self):
        region = self.slide.read_region((0, 0), 0, (10, 20))
        array = region.to_array(
            ImageInfo(ImageInfo.COLORTYPE.BGR, ImageInfo.COORD.XY,
                      ImageInfo.CHANNEL.LAST))
        self.assertEqual(array.shape, (10, 20, 3))


class TestEcvlSlide(unittest.TestCase, BaseTestSlide):
    slide = EcvlSlide(IMAGE)


class TestOpenSlide(unittest.TestCase, BaseTestSlide):
    slide = OpenSlide(IMAGE)


class TestImage(unittest.TestCase):
    def test_reads_region(self):
        slide = EcvlSlide(IMAGE)
        image = slide.read_region((0, 0), 0, (256, 256))
        self.assertEqual(image.dimensions, (3, 256, 256))


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
