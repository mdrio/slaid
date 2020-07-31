import unittest
import json
from shutil import copy
import uuid
import os
import cloudpickle as pickle
from PIL import Image
import numpy as np
from slaid.renderers import BasicFeatureTIFFRenderer, PickleRenderer, JSONEncoder
from test_commons import DummySlide
from slaid.commons import Patch, Slide
from slaid.classifiers import KarolinskaFeature


class BasicFeatureTIFFRendererTest(unittest.TestCase):
    def test_render_patch(self):
        features = {KarolinskaFeature.CANCER_PERCENTAGE: 1}
        patch = Patch(DummySlide('slide', (100, 100)), (0, 0), (10, 10),
                      features)
        renderer = BasicFeatureTIFFRenderer()
        output = '/tmp/patch.tiff'
        renderer.render_patch(output, patch)
        image = Image.open(output)
        data = np.array(image)
        self.assertEqual(data.shape, (10, 10, 4))
        self.assertTrue((data[:, :, 0] == 255).all())


class PickleRendererTest(unittest.TestCase):
    def test_render(self):
        tmp_slide = uuid.uuid4().hex
        copy('data/input.tiff', tmp_slide)
        slide = Slide(tmp_slide, extraction_level=0)
        pickle_renderer = PickleRenderer()
        output = '/tmp/slide-df.pkl'
        pickle_renderer.render(output, slide)
        os.remove(tmp_slide)
        with open(output, 'rb') as f:
            pickled_slide = pickle.load(f)
        self.assertEqual(slide, pickled_slide)


class JsonRendererTest(unittest.TestCase):
    def test_np_array(self):
        array = np.zeros((10, 10))
        jsoned_array = json.dumps(array, cls=JSONEncoder)
        jsoned_array = np.array(json.loads(jsoned_array))
        self.assertTrue(np.array_equal(array, jsoned_array))


if __name__ == '__main__':
    unittest.main()
