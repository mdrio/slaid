import abc
import inspect
import sys
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

PATCH_SIZE = (256, 256)


def get_class(name, module):
    return dict(inspect.getmembers(sys.modules[module], inspect.isclass))[name]


class Tensor(abc.ABC):
    @abc.abstractmethod
    def getdata() -> np.ndarray:
        pass


class Image(abc.ABC):
    @abc.abstractproperty
    def dimensions(self) -> Tuple[int, int]:
        pass

    @abc.abstractmethod
    def to_array(self, PIL_FORMAT: bool = False) -> np.ndarray:
        pass

    @abc.abstractmethod
    def to_tensor(self):
        pass


class Slide(abc.ABC):
    def __init__(self, filename: str, extraction_level=2):
        self._filename = filename
        self._extraction_level = extraction_level
        self.patches: PatchCollection = None
        self.masks: Dict[str, np.ndarray] = {}

    def __eq__(self, other):
        return self._filename == other._filename and\
            self.features == other.features and self.patches == other.patches

    @abc.abstractproperty
    def dimensions(self) -> Tuple[int, int]:
        pass

    @abc.abstractproperty
    def dimensions_at_extraction_level(self) -> Tuple[int, int]:
        pass

    @abc.abstractproperty
    def ID(self):
        pass

    @abc.abstractmethod
    def read_region(self, location: Tuple[int, int],
                    size: Tuple[int, int]) -> Image:
        pass

    def iterate_by_patch(self, patch_size: Tuple[int, int] = None):
        dimensions = self.dimensions_at_extraction_level
        patch_size = patch_size if patch_size else PATCH_SIZE
        for y in range(0, dimensions[1], patch_size[1]):
            for x in range(0, dimensions[0], patch_size[0]):
                yield Patch(self, (x, y), patch_size)

    @abc.abstractmethod
    def get_best_level_for_downsample(self, downsample: int):
        pass

    @abc.abstractproperty
    def level_dimensions(self):
        pass

    @abc.abstractproperty
    def level_downsamples(self):
        pass


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

    def __eq__(self, other):
        def _check_features():
            res = self.features.keys() == other.features.keys()
            for f, v in self.features.items():
                if isinstance(v, np.ndarray):
                    res = res and np.array_equal(v, other.features[f])
                else:
                    res = res and v == other.features[f]
                if not res:
                    break
            return res

        return self.slide == other.slide and (self.x, self.y) == (
            other.x,
            other.y) and self.size == other.size and _check_features()


def round_to_patch(coordinates, patch_size):
    res = []
    for i, c in enumerate(coordinates):
        size = patch_size[i]
        q, r = divmod(c, size)
        res.append(size * (q + round(r / size)))
    return tuple(res)


class PatchCollection(abc.ABC):
    class Projection(abc.ABC):
        @abc.abstractmethod
        def __eq__(self, other):
            pass

        @abc.abstractmethod
        def __lt__(self, other):
            pass

        @abc.abstractmethod
        def __le__(self, other):
            pass

        @abc.abstractmethod
        def __ne__(self, other):
            pass

        @abc.abstractmethod
        def __ge__(self, other):
            pass

        @abc.abstractmethod
        def __gt__(self, other):
            pass

        @abc.abstractmethod
        def isnull(self):
            pass

        @abc.abstractmethod
        def notnull(self):
            pass

    @abc.abstractclassmethod
    def from_pandas(cls, slide: Slide, patch_size: Tuple[int, int],
                    dataframe: pd.DataFrame):
        pass

    def __init__(self,
                 slide: Slide,
                 patch_size: Tuple[int, int] = PATCH_SIZE,
                 extraction_level=""):
        self._slide = slide
        slide.patches = self
        self._patch_size = patch_size
        self._extraction_level = extraction_level

    @property
    def slide(self) -> Slide:
        return self._slide

    @property
    def extraction_level(self) -> int:
        return self._extraction_level

    @property
    def patch_size(self) -> Tuple[int, int]:
        return self._patch_size

    @abc.abstractproperty
    def dataframe(self) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def filter(
        self,
        condition: Union[str,
                         "PatchCollection.Projection"]) -> "PatchCollection":
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
    class Projection(PatchCollection.Projection):
        def __init__(self, series: pd.core.series.Series):
            self._series = series

        def __eq__(self, other):
            return PandasPatchCollection.Projection(self._series == other)

        def __lt__(self, other):
            return PandasPatchCollection.Projection(self._series < other)

        def __le__(self, other):
            return PandasPatchCollection.Projection(self._series < other)

        def __ne__(self, other):
            return PandasPatchCollection.Projection(self._series != other)

        def __ge__(self, other):
            return PandasPatchCollection.Projection(self._series >= other)

        def __gt__(self, other):
            return PandasPatchCollection.Projection(self._series > other)

        def __and__(self, other):
            return PandasPatchCollection.Projection(self._series
                                                    & other._series)

        def __or__(self, other):
            return PandasPatchCollection.Projection(self._series
                                                    | other._series)

        def isnull(self):
            return PandasPatchCollection.Projection(self._series.isnull())

        def notnull(self):
            return PandasPatchCollection.Projection(self._series.notnull())

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

    def __init__(self,
                 slide: Slide,
                 patch_size: Tuple[int, int] = PATCH_SIZE,
                 extraction_level=2):
        super().__init__(slide, patch_size, extraction_level)
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
        return PandasPatchCollection.Projection(self._dataframe[key])

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
        return [c for c in self._dataframe.columns]

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

    def filter(
        self,
        condition: Union[str,
                         "PatchCollection.Projection"]) -> "PatchCollection":
        return PandasPatchCollection.from_pandas(
            self.slide,
            self.patch_size, self._dataframe.query(condition)) if isinstance(
                condition, str) else self._loc[condition._series]

    def update(self, other: "PandasPatchCollection"):
        self.dataframe.update(other.dataframe)

    def merge(self, other_collection: 'PandasPatchCollection'):
        self._dataframe = self.dataframe.merge(other_collection.dataframe,
                                               'left',
                                               on=['y', 'x']).set_index(
                                                   self._dataframe.index)

    def __eq__(self, other):
        return self._dataframe.equals(other._dataframe)
