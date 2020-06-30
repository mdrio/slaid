from openslide import Slide
from typing import Tuple, List
from tensorflow.keras.models import Model
from collections import namedtuple
import numpy as np  # linear algebra
from PIL import Image
import random

PatchSize = namedtuple('PatchSize', ['x', 'y'])


class Patch(object):
    pass


class InvalidSlide(Exception):
    pass


class PatchGenerator(object):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 model: Model,
                 patch_size: PatchSize,
                 class_threshold: float = 0.05
                 ):

        self._input_shape = input_shape
        self._model = model
        self._patch_size = patch_size
        self._class_threshold = class_threshold

    def _predict_thumbnails(self, slide: Slide, mask: Slide):
        lev = slide.get_best_level_for_downsample(16)
        lev_dim = slide.level_dimensions[lev]
        slide_thumb = slide.read_region(
            location=(0, 0),
            level=lev,
            size=lev_dim
        )
        slide_thumb = slide_thumb.resize((128, 128)).\
            convert('RGB')
        # check if mask is correct, otherwise skip slide
        mask_thumb = mask.read_region(location=(0, 0), level=lev, size=lev_dim)
        return slide_thumb, mask_thumb

    def _check_valid_mask(mask_thumb):
        pal = np.array(list(map(lambda c: c[1], mask_thumb.getcolors())))
        if (pal[:, 3].min() < 255):
            raise InvalidSlide()
        v = pal[:, 0]
        v.sort()
        if (v[0] != 0 or v[v.size-1] < v.size-1):
            raise InvalidSlide()

    def _compute_tissue(self, mask, slide_thumb):
        xx = mask.dimensions[0] // self._patch_size.x
        yy = mask.dimensions[1] // self._patch_size.y

        ar_mask = np.array(slide_thumb.getdata()).\
            reshape([1] + self._input_shape)
        ar_mask = (ar_mask / 255.)  # normalize
        ar_mask = self._model.predict(ar_mask)
        ar_mask = ar_mask[0].argmin(axis=-1).astype(int)
        mask = Image.new('L', self._input_shape[0:2])
        mask.putdata(ar_mask.flatten())
        # resize and use to extract patches
        mask = mask.resize((xx, yy), resample=Image.BILINEAR)
        tissue = [(x, y) for x in range(xx) for y in range(yy) if
                  mask.getpixel((x, y)) > 0]
        random.shuffle(tissue)  # randomly permute the elements
        return tissue

    def _generate_patch(self, slide, mask, num, x, y):
        m_patch = mask.read_region(
            location=(x*self._patch_size.x, y*self._patch_size.y),
            level=0,
            size=(self._patch_size.y, self._patch_size.y)
        )
        m_patch = m_patch.getchannel(0)
        colors = m_patch.getcolors()
        colors.sort(key=lambda x: -x[0])  # sort by frequency
        cow = np.zeros(6)
        for col in colors:
            cow[col[1]] += col[0]
        cow /= cow.sum()
        # consider background and gleasons 3,4,5
        cow = np.append(cow[0], cow[3:])
        if (cow[0] > .15):
            cow[0] = 1
        cow[cow >= self._class_threshold] = 1
        cow[cow < self._class_threshold] = 0
        cow = cow.reshape((1, 4)).astype(int)
        # save patches and labels
        patch = slide.read_region(
            location=(x*self._patch_size.x, y*self._patch_size.y),
            level=0,
            size=(self._patch_size.x, self._patch_size.y)
        )
        return patch.convert('RGB')

    def extract_tissue_patches(self, slide: Slide, mask: Slide) -> List[Patch]:
        patches = []

        slide_thumb, mask_thumb = self._predict_thumbnails(slide, mask)
        self._check_valid_mask(mask_thumb)
        tissue = self._compute_tissue(slide, mask_thumb)

        for (num, (x, y)) in enumerate(tissue):
            patches.append(self._generate_patch(slide, mask, num, x, y))
        return patches

    def classify_patch(self, patch, model):
        pass
