import os
import pickle

import numpy as np
from PIL import Image
from test_classifiers import GreenIsTissueModel

import slaid.classifiers as cl
from slaid.commons.ecvl import Slide
from slaid.renderers import (BasicFeatureTIFFRenderer, convert_to_heatmap,
                             to_json)

DIR = os.path.dirname(os.path.realpath(__file__))


def generate_row_first_row_cancer(filename, slide_size, patch_size):
    w, h = slide_size[::-1]
    data = np.zeros((h, w, 3), dtype=np.uint8)
    data[0:patch_size[1], 0:w] = [2, 255, 0]
    img = Image.fromarray(data, 'RGB')
    img.save(filename)


def main():
    with open(os.path.join(DIR, 'tests/data/random-model.pkl'), 'rb') as f:
        model = pickle.load(f)
    patch_size = (256, 256)
    slide_filename = os.path.join(DIR, 'tests/data/PH10023-1.thumb.tif')
    slide = Slide(slide_filename, patch_size=patch_size, extraction_level=0)

    json_filename = os.path.join(DIR, 'test.json')
    tiff_filename = os.path.join(DIR, 'test.tiff')

    tissue_classifier = cl.BasicClassifier(GreenIsTissueModel(), 'tissue')

    cancer_classifier = cl.BasicClassifier(model, 'cancer')

    print('tissue classification')

    tissue_classifier.classify(slide, include_mask=True)
    print('cancer classification')
    cancer_classifier.classify(slide, slide.patches['tissue'] > 0.5)

    to_json(slide, json_filename)

    renderer = BasicFeatureTIFFRenderer(convert_to_heatmap)

    print('rendering...')
    renderer.render(tiff_filename, slide)

    output_image = Image.open(tiff_filename)
    output_data = np.array(output_image)
    assert (output_data[0:patch_size[1], :, 0] == 255).all()
    assert (output_data[patch_size[1]:, :, 0] == 0).all()


if __name__ == '__main__':
    main()
