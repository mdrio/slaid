import unittest
import json
import pickle
from PIL import Image
import numpy as np
from slaid.renderers import BasicFeatureTIFFRenderer, PickleRenderer, JSONEncoder
from test_commons import DummySlide
from slaid.commons import Patch
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
        data = np.random.randint(0, 255, (10, 10, 4))
        slide = DummySlide('slide', (100, 100))
        pickle_renderer = PickleRenderer()
        output = '/tmp/slide-df.pkl'
        pickle_renderer.render(output, slide)
        with open(output, 'rb') as f:
            pickled_df = pickle.load(f)
        self.assertTrue(slide.patches.dataframe.equals(pickled_df))


class JsonRendererTest(unittest.TestCase):
    def test_np_array(self):
        array = np.zeros((10, 10))
        jsoned_array = json.dumps(array, cls=JSONEncoder)
        jsoned_array = np.array(json.loads(jsoned_array))
        self.assertTrue(np.array_equal(array, jsoned_array))


if __name__ == '__main__':
    unittest.main()
