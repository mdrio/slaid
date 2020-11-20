import dask.array as da
import numpy as np
import pytest
import tiledb


@pytest.fixture
def slide_with_mask():
    from slaid.commons.ecvl import Slide

    def _slide_with_mask(create_array_func):
        slide = Slide('tests/data/PH10023-1.thumb.tif')
        mask = create_array_func(slide.dimensions[::-1])
        slide.masks['mask'] = mask
        return mask

    return _slide_with_mask


@pytest.fixture
def array(request):
    return np.ones((10, 10))


@pytest.fixture
def dask_array(request):
    return da.ones((10, 10))


@pytest.fixture
def tiledb_path(tmp_path):
    tmp_path = str(tmp_path)
    tiledb.from_numpy(tmp_path, np.ones((10, 10)))
    with tiledb.open(tmp_path, 'w') as array:
        array.meta['extraction_level'] = 1
        array.meta['level_downsample'] = 1
        array.meta['threshold'] = 0.8
    return tmp_path
