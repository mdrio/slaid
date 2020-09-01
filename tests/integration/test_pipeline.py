import os

import numpy as np
from commons import AllOneModel

import slaid.classifiers as cl
from slaid.classifiers.eddl import Model
from slaid.commons.ecvl import create_slide
from slaid.renderers import (BasicFeatureTIFFRenderer, convert_to_heatmap,
                             to_json)

DIR = os.path.dirname(os.path.realpath(__file__))


def main():

    #  with open('slaid/models/extract_tissue_LSVM-1.0.pickle', 'rb') as f:
    #      tissue_model = pickle.load(f)

    tissue_model = Model('slaid/models/extract_tissue_eddl-1.0.0.bin', False)
    patch_size = (256, 256)
    slide_filename = 'tests/data/PH10023-1.thumb.tif'
    slide = create_slide(slide_filename, 0, patch_size)

    json_filename = '/tmp/test.json'
    mask_filename = 'PH10023-1-mask'
    tiff_filename = '/tmp/test.tiff'

    tissue_classifier = cl.BasicClassifier(tissue_model, 'tissue')
    cancer_classifier = cl.BasicClassifier(AllOneModel(), 'cancer')

    print('tissue classification')
    tissue_classifier.classify(slide, include_mask=True)

    print('cancer classification')
    cancer_classifier.classify(slide, slide.patches['tissue'] > 0.5)

    print('to_json')
    to_json(slide, json_filename)

    renderer = BasicFeatureTIFFRenderer(convert_to_heatmap)

    print('rendering...')
    #  renderer.render(tiff_filename, slide, 'tissue')

    np.save(mask_filename, slide.masks['tissue'])


if __name__ == '__main__':
    main()
