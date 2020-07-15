import unittest
from classifiers import PatchFeature, PatchFeatureCollection
from commons import Patch
from test_commons import DummySlide


class TestPatchFeatureCollection(unittest.TestCase):
    def test_sort(self):
        slide = DummySlide('slide', (400, 200))
        patch_size = (100, 100)
        features = [
            PatchFeature(Patch(slide, (0, 0), patch_size), {}),
            PatchFeature(Patch(slide, (0, 100), patch_size), {}),
            PatchFeature(Patch(slide, (0, 200), patch_size), {}),
            PatchFeature(Patch(slide, (0, 300), patch_size), {}),
            PatchFeature(Patch(slide, (
                0,
                0,
            ), patch_size), {}),
            PatchFeature(Patch(slide, (
                0,
                100,
            ), patch_size), {}),
            PatchFeature(Patch(slide, (
                0,
                200,
            ), patch_size), {}),
            PatchFeature(Patch(slide, (
                0,
                300,
            ), patch_size), {}),
        ]

        reversed_features = list(reversed(features))
        collection = PatchFeatureCollection(slide, patch_size,
                                            reversed_features)
        collection.sort()
        self.assertEqual(features, collection.features)


if __name__ == '__main__':
    unittest.main()
