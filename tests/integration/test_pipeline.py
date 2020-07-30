import json
import pickle
import glob
import numpy as np
import os
from PIL import Image
import slaid.classifiers as cl
from slaid.commons import Slide, Patch
from slaid.classifiers import TissueFeature
from slaid.renderers import JSONEncoder, BasicFeatureTIFFRenderer,\
    convert_to_heatmap, PickleRenderer
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

    mask = slide = Slide(slide_filename,
                         patch_size=patch_size,
                         extraction_level=0)

    tissue_classifier = cl.TissueClassifier(
        cl.BasicTissueMaskPredictor(GreenIsTissueModel()))

    cancer_classifier = cl.KarolinskaTrueValueClassifier(mask)

    print('tissue classification')

    tissue_classifier.classify(slide, include_mask_feature=True)
    print('cancer classification')
    cancer_classifier.classify(
        slide, slide.patches[cl.TissueFeature.TISSUE_PERCENTAGE] > 0.5)

    with open(json_filename, 'w') as json_file:
        json.dump(slide.patches, json_file, cls=JSONEncoder)

    renderer = BasicFeatureTIFFRenderer(convert_to_heatmap)

    print('rendering...')
    renderer.render(tiff_filename, slide)

    pickle_renderer = PickleRenderer()
    pkl_filename = '/tmp/test'
    [os.remove(f) for f in glob.glob(f'{pkl_filename}*.pkl')]

    pickled_slide_fn = '/tmp/slide.pkl'
    pickle_renderer.render(pickled_slide_fn, slide)
    os.remove(slide_filename)
    with open(pickled_slide_fn, 'rb') as f:
        pickle.load(f)

    #  for patch in slide.patches.filter(
    #          slide.patches[TissueFeature.TISSUE_MASK].notnull()):
    #      fn = f'{pkl_filename}-{patch.x}-{patch.y}.pkl'
    #      pickle_renderer.render_patch(fn, patch)
    #      assert os.path.exists(fn)
    #      with open(fn, 'rb') as f:
    #          assert isinstance(pickle.load(f), Patch)
    #          assert TissueFeature.TISSUE_MASK in patch.features

    output_image = Image.open(tiff_filename)
    output_data = np.array(output_image)
    assert (output_data[0:patch_size[1], :, 0] == 255).all()
    assert (output_data[patch_size[1]:, :, 0] == 0).all()


if __name__ == '__main__':
    main()
