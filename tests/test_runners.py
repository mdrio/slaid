import numpy as np
from slaid.runners import parse_filter
from slaid.writers import zarr as zarr_io


def test_filter_parser(slide_with_mask, tmp_path):

    path = f'{tmp_path}.zarr'
    slide = slide_with_mask(np.ones)
    condition = 'mask>2'
    zarr_io.dump(slide, path)
    res_slide, res_condition = parse_filter(f'{condition}@{path}')
    assert res_slide == slide
    assert res_condition == condition
