import json
import unittest

import numpy as np

from slaid.commons.ecvl import Slide
from slaid.commons import Mask
from slaid.renderers import to_json

#  class BasicFeatureTIFFRendererTest(unittest.TestCase):
#      def test_render_patch(self):
#          features = {'cancer': 1}
#          patch = Patch(DummySlide('slide', (100, 100)), (0, 0), (10, 10),
#                        features)
#          renderer = BasicFeatureTIFFRenderer()
#          output = '/tmp/patch.tiff'
#          renderer.render_patch(output, patch, feature='cancer')
#          image = Image.open(output)
#          data = np.array(image)
#          self.assertEqual(data.shape, (10, 10, 4))
#          self.assertTrue((data[:, :, 0] == 255).all())
#


class ToJsonTest(unittest.TestCase):
    def test_np_array(self):
        array = np.zeros((10, 10))
        jsoned_array = json.loads(to_json(array))
        self.assertTrue(np.array_equal(array, jsoned_array))

    def test_slide(self):
        #  given
        slide = Slide('tests/data/test.tif')
        array = np.ones((10, 10))
        slide.masks['annotation'] = Mask(array, 0, 1)

        #  when
        jsoned = json.loads(to_json(slide))
        #  then
        self.assertEqual(jsoned['filename'], slide.ID)
        self.assertEqual(len(jsoned['masks']), 1)
        self.assertEqual(jsoned['masks'].keys(), {'annotation'})
        self.assertEqual(set(jsoned['masks']['annotation'].keys()),
                         {'array', 'extraction_level', 'level_downsample'})
        self.assertEqual(jsoned['masks']['annotation']['extraction_level'], 0)
        self.assertEqual(jsoned['masks']['annotation']['level_downsample'], 1)
        self.assertEqual(jsoned['masks']['annotation']['array'],
                         array.tolist())
