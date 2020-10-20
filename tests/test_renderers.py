import tifffile
import tempfile
import unittest

import numpy as np

from slaid.renderers import TIFFRenderer


class TIFFRendererTest(unittest.TestCase):
    def test_render(self):
        renderer = TIFFRenderer()
        data = np.zeros((100, 100))
        data[0, :] = 1
        with tempfile.NamedTemporaryFile(suffix='.tif') as output:
            renderer.render(data, output.name)
            data_output = tifffile.imread(output.name)
            self.assertEqual(data_output.shape, (100, 100, 2))
            self.assertTrue((data_output[0, :, :] == 255).all())

            self.assertTrue((data_output[1:, :, :] == 0).all())


if __name__ == '__main__':
    unittest.main()
