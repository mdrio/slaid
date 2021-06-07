#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import numpy as np
import pytest
import tiledb

import slaid.writers.tiledb as tiledb_io
import slaid.writers.zarr as zarr_io


@pytest.mark.skip(reason="update how mask are loaded/dumped")
def test_slide_to_tiledb(slide_with_mask, tmp_path):
    slide = slide_with_mask(np.ones)
    path = str(tmp_path)
    slide_path = os.path.join(path,
                              f'{os.path.basename(slide.filename)}.tiledb')
    tiledb_io.dump(slide, slide_path)
    assert os.path.isdir(slide_path)
    for name, mask in slide.masks.items():
        assert tiledb.array_exists(os.path.join(slide_path, name))


@pytest.mark.skip(reason="update how mask are loaded/dumped")
def test_slide_from_tiledb(slide_with_mask, tmp_path):
    slide = slide_with_mask(np.ones)
    path = str(tmp_path)
    slide_path = os.path.join(path,
                              f'{os.path.basename(slide.filename)}.tiledb')
    tiledb_io.dump(slide, slide_path)
    tiledb_slide = tiledb_io.load(slide_path)

    assert os.path.basename(slide.filename) == os.path.basename(
        tiledb_slide.filename)
    assert slide.masks == tiledb_slide.masks


@pytest.mark.parametrize('storage', [zarr_io.ZarrDirectoryStorage])
def test_slide_to_zarr(storage, slide_with_mask, tmp_path):
    slide = slide_with_mask(np.ones)
    path = str(tmp_path)
    slide_path = os.path.join(path, f'{os.path.basename(slide.filename)}.zarr')
    storage.dump(slide, slide_path)
    res = storage.load(slide_path)
    assert res == slide


@pytest.mark.parametrize('storage', [zarr_io.ZarrDirectoryStorage])
def test_checks_zarr_path_has_masks(storage, slide_with_mask, tmp_path):
    slide = slide_with_mask(np.ones)
    path = str(tmp_path)
    slide_path = os.path.join(path, f'{os.path.basename(slide.filename)}.zarr')
    storage.dump(slide, slide_path)
    assert storage.mask_exists(slide_path, 'mask')
