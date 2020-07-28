import json
import numpy as np
import os
from PIL import Image
import classifiers as cl
from commons import Slide
from renderers import JSONEncoder, BasicFeatureTIFFRenderer,\
    karolinska_rgb_convert
from test_classifiers import GreenIsTissueModel

DIR = os.path.dirname(os.path.realpath(__file__))


def generate_row_first_row_cancer(filename, slide_size, patch_size):
    w, h = slide_size[::-1]
    data = np.zeros((h, w, 3), dtype=np.uint8)
    data[0:patch_size[1], 0:w] = [2, 255, 0]
    img = Image.fromarray(data, 'RGB')
    img.save(filename)


def main():
    slide_filename = os.path.join(DIR, 'input.tiff')
    patch_size = (256, 256)
    slide_size = (20 * patch_size[1], 10 * patch_size[0])
    generate_row_first_row_cancer(slide_filename, slide_size, patch_size)

    json_filename = os.path.join(DIR, 'test.json')
    tiff_filename = os.path.join(DIR, 'test.tiff')

    mask = slide = Slide(slide_filename, patch_size=patch_size)

    tissue_classifier = cl.TissueClassifier(
        cl.BasicTissueMaskPredictor(GreenIsTissueModel()))

    cancer_classifier = cl.KarolinskaTrueValueClassifier(mask)

    print('tissue classification')

    tissue_classifier.classify(slide, extraction_lev=0)
    print('cancer classification')
    cancer_classifier.classify(
        slide, slide.patches[cl.TissueFeature.TISSUE_PERCENTAGE] > 0.5)

    with open(json_filename, 'w') as json_file:
        json.dump(slide.patches, json_file, cls=JSONEncoder)

    renderer = BasicFeatureTIFFRenderer(karolinska_rgb_convert)

    print('rendering...')
    renderer.render(tiff_filename, slide)

    output_image = Image.open(tiff_filename)
    output_data = np.array(output_image)
    assert (output_data[0:patch_size[1], :, 0] == 255).all()
    assert (output_data[patch_size[1]:, :, 0] == 0).all()


if __name__ == '__main__':
    main()
