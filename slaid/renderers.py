import abc
import json
from typing import Any, Callable, Dict, List, Union

import numpy as np
from tifffile import imwrite

from slaid.commons import Patch, Slide


class Renderer(abc.ABC):
    @abc.abstractmethod
    def render(
        self,
        filename: str,
        slide: Slide,
    ):
        pass

    @abc.abstractmethod
    def render_patch(
        self,
        patch: Patch,
        filename: str,
    ):
        pass


def convert_to_heatmap(patches: List[Patch], feature: str) -> np.array:
    def _rgb_convert(patch: Patch) -> np.array:
        cancer_percentage = patch.features[feature]
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

    def render(self,
               filename: str,
               slide: Slide,
               feature: str,
               one_file_per_patch: bool = False):
        if one_file_per_patch:
            raise NotImplementedError()
        shape = slide.dimensions_at_extraction_level
        imwrite(filename,
                self._rgb_convert(slide.patches, feature),
                dtype='uint8',
                shape=(shape[1], shape[0], 4),
                photometric='rgb',
                tile=slide.patches.patch_size,
                extrasamples=('ASSOCALPHA', ))

    def render_patch(
        self,
        filename: str,
        patch: Patch,
        feature: str,
    ):
        data = list(self._rgb_convert([patch], feature))[0]
        imwrite(filename,
                data,
                photometric='rgb',
                extrasamples=('ASSOCALPHA', ))


class BaseJSONEncoder(abc.ABC):
    @abc.abstractproperty
    def target(self):
        pass

    def encode(self, obj: Any):
        pass


class PatchJSONEncoder(BaseJSONEncoder):
    @property
    def target(self):
        return Patch

    def encode(self, patch: Patch):
        return {
            'slide': patch.slide.ID,
            'x': patch.x,
            'y': patch.y,
            'size': patch.size,
            'features': patch.features
        }


class NumpyArrayJSONEncoder(BaseJSONEncoder):
    @property
    def target(self):
        return np.ndarray

    def encode(self, array: np.ndarray):
        return array.tolist()


class Int64JSONEncoder(BaseJSONEncoder):
    @property
    def target(self):
        return np.int64

    def encode(self, int_: np.int64):
        return int(int_)


# from https://github.com/hmallen/numpyencoder
def convert_numpy_types(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):

        return int(obj)

    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)

    elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return {'real': obj.real, 'imag': obj.imag}

    elif isinstance(obj, (np.ndarray, )):
        return obj.tolist()

    elif isinstance(obj, (np.bool_)):
        return bool(obj)

    elif isinstance(obj, (np.void)):
        return None
    return obj


class SlideJSONEncoder(BaseJSONEncoder):
    @property
    def target(self):
        return Slide

    def encode(
        self,
        slide: Slide,
    ) -> Union[List, Dict]:
        dct = dict(filename=slide.ID,
                   patch_size=slide.patches.patch_size,
                   extraction_level=slide.patches.extraction_level,
                   features=[])

        for p in slide.patches:
            patch_info = dict(x=p.x, y=p.y)
            for f, v in p.features.items():
                patch_info[f] = convert_numpy_types(v)
            dct['features'].append(patch_info)
        return dct


class JSONEncoder(json.JSONEncoder):
    encoders = [
        PatchJSONEncoder(),
        NumpyArrayJSONEncoder(),
        SlideJSONEncoder(),
    ]

    def default(self, obj):
        encoded = None
        for encoder in self.encoders:
            if isinstance(obj, encoder.target):
                encoded = encoder.encode(obj)
                break
        if encoded is None:
            encoded = super().default(obj)
        return encoded


class VectorialRenderer(Renderer):
    def render(self,
               slide: Slide,
               filename: str,
               one_file_per_patch: bool = False):
        if one_file_per_patch:
            raise NotImplementedError()
        with open(filename, 'w') as json_file:
            json.dump(slide.patches, json_file, cls=JSONEncoder)


def to_json(obj: Any, filename: str = None) -> Union[str, None]:
    if filename is not None:
        with open(filename, 'w') as f:
            json.dump(obj, f, cls=JSONEncoder)
    else:
        return json.dumps(obj, cls=JSONEncoder)
