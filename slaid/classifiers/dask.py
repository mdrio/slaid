import logging
from typing import Tuple

import dask.array as da
import numpy as np
from dask import delayed
from dask.distributed import Client

from slaid.classifiers.base import BasicClassifier, PatchFilter
from slaid.commons import Mask, Slide

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('dask')


def init_client(*args, **kwargs):
    return Client(*args, **kwargs)


class Classifier(BasicClassifier):
    def classify(self,
                 slide: Slide,
                 patch_filter=None,
                 threshold: float = 0.8,
                 level: int = 2,
                 patch_size=None):
        if patch_filter is not None:
            patch_filter = PatchFilter.create(slide, patch_filter).filter
            assert patch_size is not None
        else:
            patch_filter = (lambda x: True)

        if patch_size is not None:
            rows = []
            for p in slide.patches(level, patch_size):
                if patch_filter(p):
                    patch_mask = da.from_delayed(
                        self.classify_patch(slide, (p.x, p.y), level, p.size),
                        p.size[::-1], 'uint8')
                else:
                    patch_mask = da.zeros(p.size[::-1], dtype='uint8')
                row = p.y // patch_size[1]
                try:
                    rows[row].append(patch_mask)
                except IndexError:
                    rows.append([patch_mask])
            for i, r in enumerate(rows):
                rows[i] = da.concatenate(r, axis=1)
            mask = da.concatenate(rows, axis=0)

        else:
            mask = self.classify_patch(slide, (0, 0), level,
                                       slide.level_dimensions[level],
                                       threshold)

        slide.masks[self._feature] = Mask(mask.compute(), level,
                                          slide.level_downsamples[level])

    @delayed
    def classify_patch(
        self,
        slide,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        threshold: float = 0.8,
    ) -> np.ndarray:
        image = slide.read_region(location, level, size)
        image_array = self._get_image_array(image)
        prediction = self._model.predict(image_array)
        return self._get_mask(prediction, size[::-1], threshold)
