from datetime import datetime as dt

import dask.array as da
import numpy as np
import pytest
import tiledb

from slaid.commons import ImageInfo, Mask, SlideArray
from slaid.commons.ecvl import Slide as EcvlSlide
from slaid.commons.openslide import Slide as OpenSlide
from slaid.models.eddl import load_model


@pytest.fixture
def slide_with_mask():
    from slaid.commons.ecvl import Slide

    def _slide_with_mask(create_array_func):
        slide = Slide('tests/data/PH10023-1.thumb.tif')
        array = create_array_func(slide.dimensions[::-1])
        slide.masks['mask'] = Mask(array, 1, 1, dt.now(), False)
        return slide

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


@pytest.fixture
def slide_path():
    return 'tests/data/PH10023-1.thumb.tif'


@pytest.fixture
def model_all_ones_path():
    return 'tests/models/all_one_by_patch.pkl'


@pytest.fixture
def patch_path(request):
    return 'tests/data/patch.tif'


@pytest.fixture(params=[OpenSlide, EcvlSlide])
def slide_reader(request):
    return request.param


@pytest.fixture
def patch_tissue_mask(request):
    return np.load('tests/data/tissue_mask_prob.npy')


@pytest.fixture
def tissue_model():
    return load_model(
        'slaid/resources/models/tissue_model-extract_tissue_eddl_1.1.bin')


@pytest.fixture
def slide_array():
    return SlideArray(
        np.arange(16 * 3).reshape(3, 4, 4), ImageInfo('bgr', 'yx', 'first'))
