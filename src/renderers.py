import abc
import json
from tifffile import imwrite
import numpy as np
from typing import List, Tuple, Callable
from commons import Patch, PatchCollection, PATCH_SIZE, Slide
from classifiers import KarolinskaFeature


class Renderer(abc.ABC):
    @abc.abstractmethod
    def render(self, filename: str, patches: List[Patch]):
        pass


def karolinska_rgb_convert(patches: PatchCollection) -> np.array:
    for patch in patches:
        cancer_percentage = patch.features[KarolinskaFeature.CANCER_PERCENTAGE]
        cancer_percentage = 0 if cancer_percentage is None\
             else cancer_percentage
        mask_value = int(round(cancer_percentage * 255))
        data = (mask_value, 0, 0, 255) if cancer_percentage > 0 else (0, 0, 0,
                                                                      0)
        yield np.full(patch.size + (4, ), data, 'uint8')


class BasicFeatureTIFFRenderer(Renderer):
    def __init__(
        self,
        rgb_convert: Callable = karolinska_rgb_convert,
        shape: Tuple[int, int] = PATCH_SIZE,
    ):
        self._shape = shape
        self._rgb_convert = rgb_convert

    def render(self, filename: str, slide: Slide):
        imwrite(filename,
                self._rgb_convert(slide.patches),
                dtype='uint8',
                shape=(self._shape[1], self._shape[0], 4),
                photometric='rgb',
                tile=slide.patches.patch_size,
                extrasamples=('ASSOCALPHA', ))


class JSONEncoder(json.JSONEncoder):
    def _encode_patch(self, patch: Patch):

        return {
            'slide': patch.slide.ID,
            'x': patch.x,
            'y': patch.y,
            'size': patch.size,
            'features': patch.features
        }

    def default(self, obj):
        if isinstance(obj, Patch):
            return self._encode_patch(obj)
        elif isinstance(obj, PatchCollection):
            return [self._encode_patch(p) for p in obj]
        return super().default(obj)


class VectorialRenderer(Renderer):
    def render(self, slide: Slide, filename: str):
        with open(filename, 'w') as json_file:
            json.dump(slide.patches, json_file, cls=JSONEncoder)
