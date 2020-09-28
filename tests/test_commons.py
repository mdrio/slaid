import unittest

from commons import DummySlide

from slaid.commons import PandasPatchCollection, Patch, Slide, round_to_patch
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

    def test_file_not_exists(self):
        with self.assertRaises(FileNotFoundError):
            self.slide_cls('path/to/file')


#  class TestOpenSlide(unittest.TestCase, TestSlide):
#      slide = OpenSlide(IMAGE, extraction_level=0)


class TestEcvlSlide(unittest.TestCase, TestSlide):
    slide = EcvlSlide(IMAGE)
    slide_cls = EcvlSlide


class TestImage(unittest.TestCase):
    def test_to_PIL(self):
        slide = EcvlSlide(IMAGE)
        image = slide.read_region((0, 0), 0, (256, 256))
        self.assertEqual(image.dimensions, (3, 256, 256))


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
        coordinates = (257, 256)
        patch_size = (256, 256)
        res = round_to_patch(coordinates, patch_size)
        self.assertEqual(res, (256, 256))

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
