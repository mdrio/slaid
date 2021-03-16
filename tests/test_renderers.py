import tempfile
import unittest

import numpy as np
import tifffile

from slaid.renderers import TiffRenderer


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
