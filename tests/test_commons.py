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
            data = self.data[location[0]:location[0] + size[0],
                             location[1]:location[1] + size[1]]
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


if __name__ == '__main__':
    unittest.main()
