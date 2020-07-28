import unittest
from typing import Tuple
from PIL import Image
from commons import Patch, Slide, round_to_patch,\
    PatchCollection, PandasPatchCollection


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
                 data=None,
                 patches: PatchCollection = None,
                 patch_size: Tuple[int, int] = (256, 256)):

        self._id = ID
        self.size = size
        self.best_level_for_downsample = best_level_for_downsample
        self._level_dimensions = DummySlide.DummyIndexable(size)
        self._level_downsample = DummySlide.DummyIndexable(level_downsample)
        self.data = data
        self.features = {}
        self._patches = patches or PandasPatchCollection(self, patch_size)
        self.patch_size = patch_size

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

    def __len__(self):
        return self.size[0] * self.size[1] // (self.patch_size[0] *
                                               self.patch_size[1])


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


class TestRoundToPatch(unittest.TestCase):
    def test_round_0(self):
        coordinates = (0, 0)
        patch_size = (256, 256)
        res = round_to_patch(coordinates, patch_size)
        self.assertEqual(res, (0, 0))

    def test_round_1(self):
        coordinates = (256, 256)
        patch_size = (256, 256)
        res = round_to_patch(coordinates, patch_size)
        self.assertEqual(res, (256, 256))

    def test_round_down(self):
        coordinates = (513, 256)
        patch_size = (256, 256)
        res = round_to_patch(coordinates, patch_size)
        self.assertEqual(res, (512, 256))

    def test_round_up(self):
        coordinates = (511, 256)
        patch_size = (256, 256)
        res = round_to_patch(coordinates, patch_size)
        self.assertEqual(res, (512, 256))


class TestPandasPatchCollection(unittest.TestCase):
    def setUp(self):
        self.slide_size = (200, 100)
        self.slide = DummySlide('slide', self.slide_size)
        self.patch_size = (10, 10)
        self.collection = PandasPatchCollection(self.slide, self.patch_size)

    def test_init(self):
        self.assertEqual(
            len(self.collection), self.slide_size[0] * self.slide_size[1] /
            (self.patch_size[0] * self.patch_size[1]))

    def test_iteration(self):
        x = y = counter = 0
        for patch in self.collection:
            self.assertEqual(patch.x, x)
            self.assertEqual(patch.y, y)
            x = (x + self.patch_size[0]) % self.slide_size[0]
            if x == 0:
                y += self.patch_size[1]
            counter += 1

        self.assertEqual(
            counter, self.slide_size[0] * self.slide_size[1] /
            (self.patch_size[0] * self.patch_size[1]))

    def test_get_item(self):
        coordinates = (190, 90)
        patch = self.collection.get_patch(coordinates)
        self.assertTrue(isinstance(patch, Patch))
        self.assertEqual(patch.x, coordinates[0])
        self.assertEqual(patch.y, coordinates[1])

    def test_update_patch(self):
        coordinates = (190, 90)
        self.collection.update_patch(coordinates=coordinates,
                                     features={
                                         'test': 1,
                                         'test2': 2
                                     })
        self.assertEqual(len(self.collection), 200)
        patch = self.collection.get_patch(coordinates)
        self.assertEqual(patch.x, coordinates[0])
        self.assertEqual(patch.y, coordinates[1])
        self.assertEqual(patch.features['test'], 1)
        self.assertEqual(patch.features['test2'], 2)

    def test_merge(self):
        for i, p in enumerate(self.collection):
            self.collection.update_patch(patch=p, features={'feature': i})
        filtered_collection = self.collection.filter(
            self.collection['feature'] > 0)
        self.assertEqual(len(filtered_collection), len(self.collection) - 1)

        filtered_collection.update_patch(coordinates=(10, 10),
                                         features={
                                             'feature2': -1,
                                         })
        #  self.collection.update(filtered_collection)
        #  self.assertEqual(
        #      self.collection.get_patch((10, 10)).features['feature'], -1)

        self.collection.merge(filtered_collection)
        self.assertEqual(
            self.collection.get_patch((10, 10)).features['feature2'], -1)

    def test_filter(self):
        for i, p in enumerate(self.collection):
            self.collection.update_patch(patch=p, features={
                'feature': i,
            })
        filtered_collection = self.collection.filter(
            self.collection['feature'] > 0)

        for p in filtered_collection:
            self.assertTrue(p.features['feature'] > 0)

    def test_filter_textual(self):
        for i, p in enumerate(self.collection):
            self.collection.update_patch(patch=p, features={
                'feature': i,
            })
        filtered_collection = self.collection.filter('feature > 0')

        for p in filtered_collection:
            self.assertTrue(p.features['feature'] > 0)

    def test_filter_and_condition(self):
        for i, p in enumerate(self.collection):
            self.collection.update_patch(patch=p,
                                         features={
                                             'feature': i,
                                             'feature2': i
                                         })
        filtered_collection = self.collection.filter(
            (self.collection['feature'] > 0)
            & (self.collection['feature2'] > 1))

        for p in filtered_collection:
            self.assertTrue(p.features['feature'] > 0)
            self.assertTrue(p.features['feature2'] > 1)

    def test_features(self):
        self.assertEqual(self.slide.patches.features, [])
        for i, p in enumerate(self.collection):
            self.collection.update_patch(patch=p,
                                         features={
                                             'feature': i,
                                             'feature2': i
                                         })

        features = list(self.collection.features)
        features.sort()
        self.assertEqual(features, ['feature', 'feature2'])


if __name__ == '__main__':
    unittest.main()
