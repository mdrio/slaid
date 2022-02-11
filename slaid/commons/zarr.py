from typing import Tuple

import zarr

from slaid.commons.base import ArrayFactory as BaseArrayFactory


class ArrayFactory(BaseArrayFactory):

    def __init__(self, store=None):
        self._store = store

    def empty(self, shape: Tuple[int, int], dtype: str):
        return zarr.creation.empty(shape, dtype=dtype, store=self._store)

    def zeros(self, shape: Tuple[int, int], dtype: str):
        return zarr.creation.zeros(shape, dtype=dtype, store=self._store)
