import logging

import dask.array as da
import numpy as np
from dask import delayed
from dask.distributed import Client

from slaid.classifiers.base import BasicClassifier
from slaid.commons import Mask
from slaid.models import Model

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('dask')


def init_client(*args, **kwargs):
    return Client(*args, **kwargs)


class RowClassifier(BasicClassifier):
    def __init__(self, model: Model, feature: str, row_size: int):
        super().__init__(model, feature)
        self._row_size = row_size

    @property
    def row_size(self):
        return self._row_size

    def _classify_whole_slide(self, slide, threshold, level):

        dimensions = slide.level_dimensions[level]
        rows = []
        for i in range(0, dimensions[1], self._row_size):
            row_size = min(self._row_size, dimensions[1] - i)
            rows.append(
                da.from_delayed(self._get_area_mask(slide, (0, i),
                                                    (dimensions[0], row_size),
                                                    threshold), (
                                                        dimensions[0],
                                                        row_size,
                                                    ),
                                dtype='uint8'))

        mask = da.concatenate(rows, axis=1).transpose().compute()
        # FIXME duplicated code
        for patch in slide.patches:
            patch_mask = mask[patch.y:patch.y + patch.size[1],
                              patch.x:patch.x + patch.size[0]]

        slide.masks[self._feature] = Mask(mask, level)

    @delayed
    def _get_area_mask(self, slide, location, size, threshold):
        image = slide.read_region(location, size)
        image_array = image.to_array(True)
        shape = image_array.shape[:2]
        n_px = shape[0] * shape[1]
        image_array = image_array[:, :, :3].reshape(n_px, 3)
        prediction = self._model.predict(image_array)
        mask = prediction.reshape(shape)
        mask[mask >= threshold] = 1
        mask[mask < threshold] = 0
        mask = np.array(mask, dtype='uint8')
        mask = mask.transpose()
        return mask
