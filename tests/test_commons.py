import unittest

import numpy as np
import pytest
import tiledb

from slaid.commons import Mask, Slide
from slaid.commons.dask import Mask as DaskMask
from slaid.commons.ecvl import Slide as EcvlSlide
from slaid.commons.openslide import Slide as OpenSlide

IMAGE = 'tests/data/test.tif'


class BaseTestSlide:
    slide: Slide = None
    slide_cls = None

    def test_returns_dimensions(self):
        self.assertEqual(self.slide.dimensions, (512, 1024))

    def test_converts_to_array_xy_channel_first(self):
        region = self.slide.read_region((0, 0), 0, (10, 20))
        array = region.to_array('bgr', 'xy', 'first')
        self.assertEqual(array.shape, (3, 10, 20))

    def test_converts_to_array_yx_channel_first(self):
        region = self.slide.read_region((0, 0), 0, (10, 20))
        array = region.to_array('bgr', 'yx', 'first')
        self.assertEqual(array.shape, (3, 20, 10))

    def test_converts_to_array_xy_channel_last(self):
        region = self.slide.read_region((0, 0), 0, (10, 20))
        array = region.to_array('bgr', 'xy', 'last')
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


if __name__ == '__main__':
    unittest.main()
