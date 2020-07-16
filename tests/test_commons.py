import unittest
from typing import Tuple
from PIL import Image
from commons import Patch, Slide


class DummySlide(Slide):
    class DummyIndexable:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, k):
            return self.value

    def __init__(self,
                 ID: str,
                 size: Tuple[int, int],
                 image: Image = None,
                 best_level_for_downsample: int = 2,
                 level_downsample: int = 1):
        self._id = ID
        self.size = size
        self.image = image
        self.best_level_for_downsample = best_level_for_downsample
        self._level_dimensions = DummySlide.DummyIndexable(size)
        self._level_downsample = DummySlide.DummyIndexable(level_downsample)

    @property
    def dimensions(self):
        return self.size

    def read_region(self, location: Tuple[int, int], level: int,
                    size: Tuple[int, int]):
        return self.image

    @property
    def ID(self):
        return self._id

    def get_best_level_for_downsample(self, downsample: int):
        return self.best_level_for_downsample

    @property
    def level_dimensions(self):
        return self._level_dimensions

    @property
    def level_downsamples(self):
        return self._level_downsample


class TestPatch(unittest.TestCase):
    def test_ordering_0(self):
        slide = DummySlide('slide', (400, 200))
        patch_size = (100, 100)
        patch_0 = Patch(slide, (0, 0), patch_size)
        patch_1 = Patch(slide, (0, 0), patch_size)
        self.assertTrue(patch_0 == patch_1)

    def test_ordering_1(self):
        slide = DummySlide('slide', (400, 200))
        patch_size = (100, 100)
        patch_0 = Patch(slide, (0, 0), patch_size)
        patch_1 = Patch(slide, (0, 100), patch_size)
        self.assertTrue(patch_0 < patch_1)
        self.assertFalse(patch_0 > patch_1)
        self.assertFalse(patch_0 == patch_1)

    def test_ordering_2(self):
        slide = DummySlide('slide', (400, 200))
        patch_size = (100, 100)
        patch_0 = Patch(slide, (0, 0), patch_size)
        patch_1 = Patch(slide, (100, 0), patch_size)
        self.assertTrue(patch_0 < patch_1)

    def test_ordering_3(self):
        slide = DummySlide('slide', (400, 200))
        patch_size = (100, 100)
        patch_0 = Patch(slide, (0, 0), patch_size)
        patch_1 = Patch(slide, (100, 100), patch_size)
        self.assertTrue(patch_0 < patch_1)


if __name__ == '__main__':
    unittest.main()
