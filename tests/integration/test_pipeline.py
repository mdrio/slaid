import os

import numpy as np
from commons import AllOneModel

import slaid.classifiers as cl
from slaid.models.eddl import Model
from slaid.commons.ecvl import Slide
from slaid.renderers import (BasicFeatureTIFFRenderer, convert_to_heatmap,
                             to_json)

DIR = os.path.dirname(os.path.realpath(__file__))


def main():

    #  with open('slaid/models/extract_tissue_LSVM-1.0.pickle', 'rb') as f:
    #      tissue_model = pickle.load(f)

    tissue_model = Model(
        'slaid/resources/models/extract_tissue_eddl-1.0.0.bin', False)
    slide_filename = 'tests/data/PH10023-1.thumb.tif'
    slide = Slide(slide_filename)

    json_filename = '/tmp/test.json'
    mask_filename = 'PH10023-1-mask'
    #  tiff_filename = '/tmp/test.tiff'

    tissue_classifier = cl.BasicClassifier(tissue_model, 'tissue')
    cancer_classifier = cl.BasicClassifier(AllOneModel(), 'cancer')

    print('tissue classification')
    mask = tissue_classifier.classify(slide)
    slide.masks['tissue'] = mask

    print('cancer classification')
    mask = cancer_classifier.classify(slide,
                                      patch_filter='tissue > 0.5',
                                      patch_size=(256, 256))

    slide.masks['cancer'] = mask
    print('to_json')
    to_json(slide, json_filename)

    #  renderer = BasicFeatureTIFFRenderer(convert_to_heatmap)

    print('rendering...')
    #  renderer.render(tiff_filename, slide, 'tissue')

    np.save(mask_filename, slide.masks['tissue'])


if __name__ == '__main__':
    main()
