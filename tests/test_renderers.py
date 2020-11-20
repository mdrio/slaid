import os
import tempfile
import unittest

import numpy as np
import tifffile

from slaid.renderers import TiffRenderer, from_tiledb, to_tiledb
import tiledb


class TestTiffRenderer(unittest.TestCase):
    def test_render_grayscale(self):
        renderer = TiffRenderer(rgb=False)
        data = np.zeros((100, 100))
        data[0, :] = 1
        with tempfile.NamedTemporaryFile(suffix='.tif') as output:
            renderer.render(data, output.name)
            data_output = tifffile.imread(output.name)
            self.assertEqual(data_output.shape, (100, 100, 2))

            self.assertTrue((data_output[0, :, 0] == 255).all())
            self.assertTrue((data_output[0, :, 1] == 255).all())

            self.assertTrue((data_output[1:, :, :, ] == 0).all())

    def test_render_rgb(self):
        renderer = TiffRenderer()
        data = np.zeros((100, 100))
        data[0, :] = 1
        with tempfile.NamedTemporaryFile(suffix='.tif') as output:
            renderer.render(data, output.name)
            data_output = tifffile.imread(output.name)
            self.assertEqual(data_output.shape, (100, 100, 4))

            self.assertTrue((data_output[0, :, 0] == 255).all())
            self.assertTrue((data_output[0, :, 1] == 0).all())
            self.assertTrue((data_output[0, :, 2] == 0).all())
            self.assertTrue((data_output[0, :, 3] == 255).all())

            self.assertTrue((data_output[1:, :, :, ] == 0).all())


def test_slide_to_tiledb(slide_with_mask, tmp_path):
    slide = slide_with_mask(np.ones)
    path = str(tmp_path)
    path = to_tiledb(slide, path)
    assert os.path.isdir(path)
    for name, mask in slide.masks.items():
        assert tiledb.array_exists(os.path.join(path, name))


def test_slide_from_tiledb(slide_with_mask, tmp_path):
    slide = slide_with_mask(np.ones)
    path = str(tmp_path)
    path = to_tiledb(slide, path)
    tiledb_slide = from_tiledb(path)

    assert os.path.basename(slide.filename) == os.path.basename(
        tiledb_slide.filename)
    assert slide.masks == tiledb_slide.masks
