import unittest

from commons import DummySlide

from slaid.commons import Patch, Slide, convert_patch, round_to_patch
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


class TestSlidePatches(unittest.TestCase):
    def test_iterate_default(self):
        slide = DummySlide([(20, 20)], [1])
        patch_size = (10, 10)
        expected_patches_coord = [(0, 0), (10, 0), (0, 10), (10, 10)]

        patches_coord = [(p.x, p.y) for p in slide.patches(0, patch_size)]
        self.assertEqual(patches_coord, expected_patches_coord)

    def test_iterate_custom(self):
        slide = DummySlide([(20, 20)], [1])
        patch_size = (10, 10)
        start = (0, 10)
        end = slide.dimensions
        expected_patches_coord = [(0, 10), (10, 10)]

        patches_coord = [(p.x, p.y)
                         for p in slide.patches(0, patch_size, start, end)]
        self.assertEqual(patches_coord, expected_patches_coord)

    def test_iterate_patch_greater_than_batch(self):
        slide = DummySlide([(20, 20)], [1])
        patch_size = (20, 20)
        start = (0, 0)
        end = (10, 10)
        expected_patches_coord = [(0, 0)]
        expected_patch_size = (10, 10)

        patches = list(slide.patches(0, patch_size, start, end))
        patches_coord = [(p.x, p.y) for p in patches]
        self.assertEqual(patches_coord, expected_patches_coord)
        self.assertEqual(patches[0].size, expected_patch_size)

    def test_iterate_patch_equal_batch_less_than_slide(self):
        slide = DummySlide([(30, 30)], [1])
        patch_size = (20, 20)
        start = (0, 0)
        end = (20, 20)
        expected_patches_coord = [(0, 0)]

        patches = list(slide.patches(0, patch_size, start, end))
        patches_coord = [(p.x, p.y) for p in patches]
        self.assertEqual(patches_coord, expected_patches_coord)
        self.assertEqual(patches[0].size, patch_size)

    def test_iterate_patch_equal_batch_great_than_slide(self):
        slide = DummySlide([(5, 5)], [1])
        patch_size = (20, 20)
        start = (0, 0)
        end = (20, 20)

        expected_patches_coord = [(0, 0)]
        expected_patch_size = (5, 5)

        patches = list(slide.patches(0, patch_size, start, end))
        patches_coord = [(p.x, p.y) for p in patches]
        self.assertEqual(patches_coord, expected_patches_coord)
        self.assertEqual(patches[0].size, expected_patch_size)


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


class TestConvertPatch(unittest.TestCase):
    def test_convert_down(self):
        slide = EcvlSlide('tests/data/test.tif')
        patch = Patch(100, 100, (100, 100), slide.level_downsamples[0])
        converted_patch = convert_patch(patch, slide,
                                        slide.level_downsamples[1])
        self.assertEqual(converted_patch.x, patch.x)
        self.assertEqual(converted_patch.y, patch.y)
        self.assertEqual(converted_patch.size[0], patch.size[0] // 2)
        self.assertEqual(converted_patch.size[1], patch.size[1] // 2)

    def test_convert_up(self):
        slide = EcvlSlide('tests/data/test.tif')
        patch = Patch(100, 100, (100, 100), slide.level_downsamples[1])
        converted_patch = convert_patch(patch, slide,
                                        slide.level_downsamples[0])
        self.assertEqual(converted_patch.x, patch.x)
        self.assertEqual(converted_patch.y, patch.y)
        self.assertEqual(converted_patch.size[0], patch.size[0] * 2)
        self.assertEqual(converted_patch.size[1], patch.size[1] * 2)


if __name__ == '__main__':
    unittest.main()
