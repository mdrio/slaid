import unittest
from classifiers import PatchFeature, PatchFeatureCollection
from commons import Slide
from typing import Tuple, Dict


class DummySlide(Slide):
    def __init__(self, ID: str, size: Tuple[int, int]):
        self._id = ID
        self.size = size

    @property
    def dimensions(self):
        return self.size

    def read_region(self, location: Tuple[int, int], level: int,
                    size: Tuple[int, int]):
        raise NotImplementedError()

    @property
    def ID(self):
        return self._id


class DummyPatchFeature(PatchFeature):
    def __init__(self, x: int, y: int, size: Tuple[int, int], data: Dict):
        self._x = x
        self._y = y
        self._size = size
        self.data = data

    @property
    def size(self):
        return self._size

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __eq__(self, other):
        return self.x == other.x and self.y == self.y and\
            self.size == other.size and self.data == other.data

    def __repr__(self):
        return f'{self.x}, {self.y}, {self.size}, {self.data}'


class TestPatchFeatureCollection(unittest.TestCase):
    def test_order(self):
        slide = DummySlide('slide', (400, 200))
        patch_size = (100, 100)
        features = [
            DummyPatchFeature(0, 0, patch_size, {}),
            DummyPatchFeature(100, 0, patch_size, {}),
            DummyPatchFeature(200, 0, patch_size, {}),
            DummyPatchFeature(300, 0, patch_size, {}),
            DummyPatchFeature(0, 100, patch_size, {}),
            DummyPatchFeature(100, 100, patch_size, {}),
            DummyPatchFeature(200, 100, patch_size, {}),
            DummyPatchFeature(300, 100, patch_size, {}),
        ]

        reversed_features = list(reversed(features))
        collection = PatchFeatureCollection(slide, patch_size,
                                            reversed_features)
        collection.order_features()
        print(features)
        print(collection.features)
        self.assertEqual(features, collection.features)


if __name__ == '__main__':
    unittest.main()
