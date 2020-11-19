import dask.array as da
import numpy as np
import pytest


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


#  @pytest.fixture
#  def tiledb_path(tmp_path):
#      return tiledb.from_numpy(np.ones(10, 10))
