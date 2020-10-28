import unittest

from slaid.commons import Slide
from slaid.commons.ecvl import Slide as EcvlSlide

IMAGE = 'tests/data/test.tif'


class TestSlide:
    slide: Slide = None
    slide_cls = None

    def test_dimensions(self):
        self.assertEqual(self.slide.dimensions, (512, 1024))

    def test_to_array(self):
        region = self.slide.read_region((0, 0), 0, (256, 256))
        array = region.to_array()
        self.assertEqual(array.shape, (3, 256, 256))

    def test_to_array_as_PIL(self):
        region = self.slide.read_region((0, 0), 0, (256, 256))
        array = region.to_array(True)
        self.assertEqual(array.shape, (256, 256, 3))


class TestEcvlSlide(unittest.TestCase, TestSlide):
    slide = EcvlSlide(IMAGE)
    slide_cls = EcvlSlide


class TestImage(unittest.TestCase):
    def test_to_PIL(self):
        slide = EcvlSlide(IMAGE)
        image = slide.read_region((0, 0), 0, (256, 256))
        self.assertEqual(image.dimensions, (3, 256, 256))


if __name__ == '__main__':
    unittest.main()
