import unittest
import json
import pickle
from PIL import Image
import numpy as np
from renderers import BasicFeatureTIFFRenderer, PickleRenderer, JSONEncoder
from test_commons import DummySlide
from commons import Patch
from classifiers import KarolinskaFeature


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
    def test_render_patch(self):
        data = np.random.randint(0, 255, (10, 10, 4))
        slide = DummySlide('slide', (100, 100))
        patch = Patch(slide, (0, 0), (10, 10), features={'mask': data})
        pickle_renderer = PickleRenderer()
        output = '/tmp/patch.pkl'
        pickle_renderer.render_patch(output, patch)
        with open(output, 'rb') as f:
            pickled_patch = pickle.load(f)
        self.assertEqual(patch, pickled_patch)


class JsonRendererTest(unittest.TestCase):
    def test_np_array(self):
        array = np.zeros((10, 10))
        jsoned_array = json.dumps(array, cls=JSONEncoder)
        jsoned_array = np.array(json.loads(jsoned_array))
        self.assertTrue(np.array_equal(array, jsoned_array))


if __name__ == '__main__':
    unittest.main()
