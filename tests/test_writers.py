#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import numpy as np
import tiledb

import slaid.writers.zarr as zarr_io
import slaid.writers.tiledb as tiledb_io


def test_slide_to_tiledb(slide_with_mask, tmp_path):
    slide = slide_with_mask(np.ones)
    path = str(tmp_path)
    slide_path = tiledb_io.dump(slide, path)
    assert os.path.isdir(slide_path)
    for name, mask in slide.masks.items():
        assert tiledb.array_exists(os.path.join(slide_path, name))


def test_slide_from_tiledb(slide_with_mask, tmp_path):
    slide = slide_with_mask(np.ones)
    path = str(tmp_path)
    slide_path = tiledb_io.dump(slide, path)
    tiledb_slide = tiledb_io.load(slide_path)

    assert os.path.basename(slide.filename) == os.path.basename(
        tiledb_slide.filename)
    assert slide.masks == tiledb_slide.masks


def test_slide_to_zarr(slide_with_mask, tmp_path):
    slide = slide_with_mask(np.ones)
    path = str(tmp_path)
    slide_path = zarr_io.dump(slide, path)
    res = zarr_io.load(slide_path)
    assert res == slide


#  def test_checks_zarr_path_has_masks(slide_with_mask):
#      slide = slide_with_mask(np.ones)
#      path = str(tmp_path)
#      slide_path = to_zarr(slide, path)
#      res = from_zarr(slide_path)
#
