import abc
import json
from tifffile import imwrite
import numpy as np
from typing import List, Callable, Union
from commons import Patch, PatchCollection, Slide
from classifiers import KarolinskaFeature


class Renderer(abc.ABC):
    @abc.abstractmethod
    def render(self, filename: str, slide: Slide):
        pass

    @abc.abstractmethod
    def render_patch(self, filename: str, patch: Patch):
        pass


def convert_to_heatmap(patches: List[Patch]) -> np.array:
    def _rgb_convert(patch: Patch) -> np.array:
        cancer_percentage = patch.features[KarolinskaFeature.CANCER_PERCENTAGE]
        cancer_percentage = 0 if cancer_percentage is None\
             else cancer_percentage
        mask_value = int(round(cancer_percentage * 255))
        return (mask_value, 0, 0, 255) if cancer_percentage > 0 else (0, 0, 0,
                                                                      0)

    for patch in patches:
        data = _rgb_convert(patch)
        yield np.full(patch.size + (4, ), data, 'uint8')


class BasicFeatureTIFFRenderer(Renderer):
    def __init__(
        self,
        rgb_convert: Callable = None,
    ):
        self._rgb_convert = rgb_convert or convert_to_heatmap

    def render(self, filename: str, slide: Slide):
        shape = slide.dimensions
        imwrite(filename,
                self._rgb_convert(slide.patches),
                dtype='uint8',
                shape=(shape[1], shape[0], 4),
                photometric='rgb',
                tile=slide.patches.patch_size,
                extrasamples=('ASSOCALPHA', ))

    def render_patch(self, filename: str, patch: Patch):
        data = list(self._rgb_convert([patch]))[0]
        imwrite(filename,
                data,
                photometric='rgb',
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
