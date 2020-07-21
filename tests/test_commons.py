import unittest
from typing import Tuple
from PIL import Image
from commons import Slide


class DummySlide(Slide):
    class DummyIndexable:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, k):
            return self.value

    def __init__(self,
                 ID: str,
                 size: Tuple[int, int],
                 best_level_for_downsample: int = 1,
                 level_downsample: int = 1,
                 data=None):
        self._id = ID
        self.size = size
        self.best_level_for_downsample = best_level_for_downsample
        self._level_dimensions = DummySlide.DummyIndexable(size)
        self._level_downsample = DummySlide.DummyIndexable(level_downsample)
        self.data = data

    @property
    def dimensions(self):
        return self.size

    def read_region(self, location: Tuple[int, int], level: int,
                    size: Tuple[int, int]):
        if self.data is None:
            return Image.new('RGB', size)
        else:
            data = self.data[location[1]:location[1] + size[1],
                             location[0]:location[0] + size[0]]
            mask = Image.fromarray(data, 'RGB')
            return mask

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


class TestSlide(unittest.TestCase):
    def test_iterate(self):
        patch_size = (256, 256)
        slide_size = (1024, 512)
        slide = DummySlide('slide', slide_size)
        patches = list(slide.iterate_by_patch(patch_size))
        self.assertEqual(
            len(patches),
            slide_size[0] * slide_size[1] / (patch_size[0] * patch_size[1]))

        expected_coordinates = [(0, 0), (256, 0), (512, 0), (768, 0), (0, 256),
                                (256, 256), (512, 256), (768, 256)]
        real_coordinates = [(p.x, p.y) for p in patches]
        self.assertEqual(real_coordinates, expected_coordinates)


if __name__ == '__main__':
    unittest.main()
