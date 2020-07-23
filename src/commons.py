import abc
from openslide import OpenSlide, open_slide, OpenSlideUnsupportedFormatError
from typing import Tuple, Dict, Any, List
import os
import sys
import json
import pandas as pd
from collections import defaultdict, OrderedDict
import inspect


def get_class(name, module):
    return dict(inspect.getmembers(sys.modules[module], inspect.isclass))[name]


class Slide:
    def __init__(self,
                 filename: str,
                 features: Dict = None,
                 patches: 'PatchCollection' = None,
                 patch_size: Tuple[int, int] = (256, 256)):
        self._filename = filename

        try:
            self._slide = OpenSlide(filename)
        except OpenSlideUnsupportedFormatError:
            self._slide = open_slide(filename)
        self.features = features or {}
        if patches is None and patch_size:
            self._patches = patches or PandasPatchCollection(self, patch_size)
        else:
            self._patches = patches

    @property
    def patches(self):
        return self._patches

    @property
    def dimensions(self):
        return self._slide.dimensions

    @property
    def ID(self):
        return os.path.basename(self._filename)

    def read_region(self, location: Tuple[int, int], level: int,
                    size: Tuple[int, int]):
        return self._slide.read_region(location, level, size)

    def __getstate__(self):
        return self._filename

    def __setstate__(self, filename):
        self.__init__(filename)

    def iterate_by_patch(self, patch_size: Tuple[int, int] = None):
        patch_size = patch_size if patch_size else (256, 256)
        for y in range(0, self.dimensions[1], patch_size[1]):
            for x in range(0, self.dimensions[0], patch_size[0]):
                yield Patch(self, (x, y), patch_size)

    def get_best_level_for_downsample(self, downsample: int):
        return self._slide.get_best_level_for_downsample(downsample)

    @property
    def level_dimensions(self):
        return self._slide.level_dimensions

    @property
    def level_downsamples(self):
        return self._slide.level_downsamples


class SlideIterator:
    def __init__(self, slide: Slide, patch_size: Tuple[int, int]):
        self._slide = slide
        self._patch_size = patch_size

    def __iter__(self):
        return self._slide.iterate_by_patch(self._patch_size)


class Patch:
    def __init__(self,
                 slide: Slide,
                 coordinates: Tuple[int, int],
                 size: Tuple[int, int],
                 features: Dict = None):
        self.slide = slide
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.features = features or {}

    def __str__(self):
        return (f'slide: {self.slide}, x: {self.x}, '
                f'y: {self.y}, size: {self.size}')


def round_to_patch(coordinates, patch_size):
    res = []
    for i, c in enumerate(coordinates):
        size = patch_size[i]
        q, r = divmod(c, size)
        res.append(size * (q + round(r / size)))
    return tuple(res)


class PatchCollection(abc.ABC):
    @abc.abstractclassmethod
    def from_pandas(cls, slide: Slide, patch_size: Tuple[int, int],
                    dataframe: pd.DataFrame):
        pass

    def __init__(self, slide: Slide, patch_size: Tuple[int, int]):
        self._slide = slide
        self._patch_size = patch_size

    @property
    def slide(self) -> Slide:
        return self._slide

    @property
    def patch_size(self) -> Tuple[int, int]:
        return self._patch_size

    @abc.abstractproperty
    def dataframe(self) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def filter(self,
               condition: "PatchCollection.Condition") -> "PatchCollection":
        pass

    @abc.abstractmethod
    def get_patch(self, coordinates: Tuple[int, int]) -> Patch:
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def update_patch(self,
                     coordinates: Tuple[int, int] = None,
                     patch: Patch = None,
                     features: Dict = None):
        pass

    @abc.abstractproperty
    def features(self) -> List[str]:
        pass

    @abc.abstractmethod
    def add_feature(self, feature: str, default_value: Any = None):
        pass

    @abc.abstractmethod
    def update(self, other_collection: 'PatchCollection'):
        pass

    @abc.abstractmethod
    def merge(self, other_collection: 'PatchCollection'):
        pass


class PandasPatchCollection(PatchCollection):
    class LocIndexer:
        def __init__(self, collection: "PandasPatchCollection"):
            self.collection = collection

        def __getitem__(self, key):
            return PandasPatchCollection.from_pandas(
                self.collection.slide, self.collection.patch_size,
                self.collection.dataframe.loc[key])

    @classmethod
    def from_pandas(cls, slide: Slide, patch_size: Tuple[int, int],
                    dataframe: pd.DataFrame):
        patch_collection = cls(slide, patch_size)
        # TODO add some assert to verify dataframe is compatible
        patch_collection._dataframe = dataframe
        return patch_collection

    def __init__(self, slide: Slide, patch_size: Tuple[int, int]):
        super().__init__(slide, patch_size)
        self._dataframe = self._init_dataframe()
        self._loc = PandasPatchCollection.LocIndexer(self)

    def _init_dataframe(self):
        data = defaultdict(lambda: [])
        for p in self._slide.iterate_by_patch(self._patch_size):
            data['y'].append(p.y)
            data['x'].append(p.x)
        df = pd.DataFrame(data, dtype=int)
        df.set_index(['y', 'x'], inplace=True)
        return df

    def __getitem__(self, key):
        return self._dataframe[key]

    def _create_patch(self, coordinates: Tuple[int, int],
                      data: Tuple) -> Patch:
        y, x = coordinates
        features = dict(data)
        return Patch(self.slide, (x, y), self.patch_size, features)

    def __iter__(self):
        for data in self._dataframe.iterrows():
            yield self._create_patch(data[0], data[1])

    def __len__(self):
        return len(self._dataframe)

    @property
    def dataframe(self):
        return self._dataframe

    def get_patch(self, coordinates: Tuple[int, int]) -> Patch:
        return self._create_patch(coordinates[::-1],
                                  self._dataframe.loc[coordinates[::-1]])

    @property
    def features(self) -> List[str]:
        return [c for c in self._dataframe.columns[2:]]

    def add_feature(self, feature: str, default_value: Any = None):
        if feature not in self.features:
            self.dataframe.insert(len(self._dataframe.columns), feature,
                                  default_value)

    def update_patch(self,
                     coordinates: Tuple[int, int] = None,
                     patch: Patch = None,
                     features: Dict = None):
        if patch:
            coordinates = (patch.x, patch.y)

        features = OrderedDict(features)
        if coordinates is None:
            raise RuntimeError('coordinates and patch cannot be None')

        missing_features = features.keys() - set(self._dataframe.columns)
        for f in missing_features:
            self.add_feature(f)
        self._dataframe.loc[coordinates[::-1],
                            list(features.keys())] = list(features.values())

    def filter(self,
               condition: "PatchCollection.Condition") -> "PatchCollection":
        return self._loc[condition]

    def update(self, other: "PandasPatchCollection"):
        self.dataframe.update(other.dataframe)

    def merge(self, other_collection: 'PandasPatchCollection'):
        self._dataframe = self.dataframe.merge(other_collection.dataframe,
                                               'left',
                                               on=['y', 'x']).set_index(
                                                   self._dataframe.index)


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
