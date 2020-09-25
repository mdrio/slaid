import logging

import dask.array as da
import numpy as np
from dask import delayed
from dask.distributed import Client

from slaid.classifiers.base import BasicClassifier
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

    def _classify_whole_slide(self, slide, patch_area, mask_threshold,
                              patch_threshold, include_mask):

        dimensions = slide.dimensions_at_extraction_level
        rows = []
        for i in range(0, dimensions[1], self._row_size):
            row_size = min(self._row_size, dimensions[1] - i)
            rows.append(
                da.from_delayed(self._get_area_mask(slide, (0, i),
                                                    (dimensions[0], row_size),
                                                    mask_threshold), (
                                                        row_size,
                                                        dimensions[0],
                                                    ),
                                dtype='uint8'))

        mask = da.concatenate(rows, axis=0).compute()
        # FIXME duplicated code
        for patch in slide.patches:
            patch_mask = mask[patch.y:patch.y + patch.size[1],
                              patch.x:patch.x + patch.size[0]]
            self._update_patch(slide, patch, patch_mask, patch_area,
                               patch_threshold)

        if include_mask:
            slide.masks[self._feature] = mask

    @delayed
    def _get_area_mask(self, slide, location, size, threshold):
        image = slide.read_region(location, size)
        image_array = image.to_array(True)
        n_px = image_array.shape[0] * image_array.shape[1]
        image_array = image_array[:, :, :3].reshape(n_px, 3)
        prediction = self._model.predict(image_array)
        mask = prediction.reshape(size)
        mask[mask >= threshold] = 1
        mask[mask < threshold] = 0
        mask = np.array(mask, dtype='uint8')
        mask = mask.transpose()
        return mask
