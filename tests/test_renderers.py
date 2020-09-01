import json
import unittest

import numpy as np
from PIL import Image
from test_commons import DummySlide

from slaid.commons import Patch
from slaid.renderers import BasicFeatureTIFFRenderer, to_json


class BasicFeatureTIFFRendererTest(unittest.TestCase):
    def test_render_patch(self):
        features = {'cancer': 1}
        patch = Patch(DummySlide('slide', (100, 100)), (0, 0), (10, 10),
                      features)
        renderer = BasicFeatureTIFFRenderer()
        output = '/tmp/patch.tiff'
        renderer.render_patch(output, patch, feature='cancer')
        image = Image.open(output)
        data = np.array(image)
        self.assertEqual(data.shape, (10, 10, 4))
        self.assertTrue((data[:, :, 0] == 255).all())


class ToJsonTest(unittest.TestCase):
    def test_np_array(self):
        array = np.zeros((10, 10))
        jsoned_array = json.loads(to_json(array))
        self.assertTrue(np.array_equal(array, jsoned_array))

    def test_slide(self):
        #  given
        slide = DummySlide('s', (10, 20), patch_size=(10, 10))
        prob = 10
        slide.patches.add_feature('prob', prob)

        #  when
        jsoned = json.loads(to_json(slide))
        #  then
        self.assertEqual(jsoned['filename'], slide.ID)
        self.assertEqual(tuple(jsoned['patch_size']), slide.patch_size)
        self.assertEqual(len(slide.patches), len(jsoned['features']))

        for f in jsoned['features']:
            self.assertEqual(len(f),
                             len(slide.patches.features) + 2)  # features +x +y
            self.assertEqual(
                slide.patches.get_patch((f['x'], f['y'])).features['prob'],
                f['prob'])
