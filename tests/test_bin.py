#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import shutil
import subprocess
import unittest

import numpy as np
import pytest
import zarr

from slaid.commons.ecvl import BasicSlide as Slide
from slaid.writers import REGISTRY

DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = '/tmp/test-slaid'
input_ = os.path.join(DIR, 'data/PH10023-1.thumb.tif')
input_basename = os.path.basename(input_)
input_basename_no_ext = 'PH10023-1.thumb'
slide = Slide(input_)
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)


def get_input_output(output):
    slide = Slide(input_)
    zarr_group = zarr.open_group(output)
    return slide, zarr_group


def _test_output(feature, output, slide, level):
    assert output.attrs['filename'] == slide.filename
    assert tuple(output.attrs['resolution']) == slide.dimensions
    assert output[feature].shape == slide.level_dimensions[level][::-1]
    assert output[feature].attrs['extraction_level'] == level
    assert output[feature].attrs[
        'level_downsample'] == slide.level_downsamples[level]


@pytest.mark.parametrize('cmd', ['serial', 'parallel'])
@pytest.mark.parametrize(
    'model',
    ['slaid/resources/models/tissue_model-extract_tissue_eddl_1.1.bin'])
def test_classifies_with_default_args(cmd, tmp_path, model):
    feature = 'tissue'
    path = str(tmp_path)
    cmd = [
        'classify.py', cmd, '-f', feature, '-m', model, '-l', '2', '-o', path,
        input_
    ]
    subprocess.check_call(cmd)
    logger.info('running cmd %s', ' '.join(cmd))
    output_path = os.path.join(path, f'{input_basename}.zarr')
    slide, output = get_input_output(output_path)

    _test_output(feature, output, slide, 2)
    assert output[feature].dtype == 'uint8'


@pytest.mark.parametrize('cmd', ['serial', 'parallel'])
@pytest.mark.parametrize(
    'model',
    ['slaid/resources/models/tissue_model-extract_tissue_eddl_1.1.bin'])
def test_classifies_with_no_round(cmd, tmp_path, model):
    path = str(tmp_path)
    feature = 'tissue'
    cmd = [
        'classify.py', cmd, '-f', feature, '-m', model, '--no-round', '-l',
        '2', '-o', path, input_
    ]
    subprocess.check_call(cmd)
    logger.info('running cmd %s', ' '.join(cmd))
    output_path = os.path.join(path, f'{input_basename}.zarr')
    slide, output = get_input_output(output_path)

    assert output[feature].dtype == 'float32'
    assert (np.array(output[feature]) <= 1).all()


#  @pytest.mark.parametrize('cmd', ['serial', 'parallel'])
#  @pytest.mark.parametrize(
#      'model',
#      ['slaid/resources/models/tissue_model-extract_tissue_eddl_1.1.bin'])
#  @pytest.mark.parametrize('storage', ['zarr', 'zarr-zip'])
#  def test_classifies_with_array_input(cmd, tmp_path, model, slide_with_mask,
#                                       storage):
#      feature = 'tissue'
#      slide = slide_with_mask(np.ones)
#      path = str(tmp_path)
#      slide_path = os.path.join(path,
#                                f'{os.path.basename(slide.filename)}.{storage}')
#      REGISTRY[storage].dump(slide, slide_path)
#
#      cmd = [
#          'classify.py',
#          cmd,
#          '-f',
#          feature,
#          '-m',
#          model,
#          '-o',
#          path,
#          slide_path,
#      ]
#      logger.info('cmd %s', ' '.join(cmd))
#      subprocess.check_call(cmd)
#      output = REGISTRY[storage].load(slide_path)
#      assert 'mask' in output.masks
#      assert feature in output.masks
#


class TestSerialEddlClassifier:
    model = 'slaid/resources/models/tissue_model-extract_tissue_eddl_1.1.bin'
    cmd = 'serial'
    feature = 'tissue'

    @staticmethod
    def _clean_output_dir():
        try:
            shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        except FileNotFoundError:
            pass

    def setUp(self):
        self._clean_output_dir()

    def teardown(self):
        self._clean_output_dir()

    def _test_output(self, output, slide, level):
        assert output.attrs['filename'] == slide.filename
        assert tuple(output.attrs['resolution']) == slide.dimensions
        assert output[
            self.feature].shape == slide.level_dimensions[level][::-1]
        assert output[self.feature].attrs['extraction_level'] == level
        assert output[self.feature].attrs[
            'level_downsample'] == slide.level_downsamples[level]

    def test_classifies_with_default_args(self, tmp_path):
        path = str(tmp_path)
        cmd = [
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            '-l', '2', '-o', path, input_
        ]
        subprocess.check_call(cmd)
        logger.info('running cmd %s', ' '.join(cmd))
        output_path = os.path.join(path, f'{input_basename}.zarr')
        slide, output = get_input_output(output_path)

        self._test_output(output, slide, 2)
        assert output[self.feature].dtype == 'uint8'

    def test_classifies_with_no_round(self, tmp_path):
        path = str(tmp_path)
        cmd = [
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            '--no-round', '-l', '2', '-o', path, input_
        ]
        subprocess.check_call(cmd)
        logger.info('running cmd %s', ' '.join(cmd))
        output_path = os.path.join(path, f'{input_basename}.zarr')
        slide, output = get_input_output(output_path)

        assert output[self.feature].dtype == 'float32'
        assert (np.array(output[self.feature]) <= 1).all()

    def test_classifies_zarr_input(self, tmp_path, slide_with_mask):
        slide = slide_with_mask(np.ones)
        path = str(tmp_path)
        slide_path = os.path.join(path,
                                  f'{os.path.basename(slide.filename)}.zarr')
        ZarrDirectoryStorage.dump(slide, slide_path)

        cmd = [
            'classify.py',
            self.cmd,
            '-f',
            self.feature,
            '-m',
            self.model,
            '-o',
            path,
            slide_path,
        ]
        logger.info('cmd %s', ' '.join(cmd))
        subprocess.check_call(cmd)
        output = ZarrDirectoryStorage.load(slide_path)
        assert 'mask' in output.masks
        assert self.feature in output.masks

    def test_classifies_custom_args(self):
        extr_level = 1
        cmd = f'classify.py {self.cmd} -m {self.model} -f {self.feature}  -l '\
            f' {extr_level}  -t 0.7  {input_} -o {OUTPUT_DIR}'
        subprocess.check_call(cmd.split())
        output_path = os.path.join(OUTPUT_DIR, f'{input_basename}.zarr')
        slide, output = get_input_output(output_path)

        self._test_output(output, slide, extr_level)

    def test_overwrites_existing_classification_output(self):
        output_path = os.path.join(OUTPUT_DIR, f'{input_basename}.zarr')
        os.makedirs(output_path)
        subprocess.check_call([
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            '--overwrite', '-o', OUTPUT_DIR, input_
        ])
        stats = os.stat(output_path)
        assert stats.st_size > 0
        slide, output = get_input_output(output_path)

        self._test_output(output, slide, 2)


class TestSerialPatchClassifier:
    model = 'tests/models/all_one_by_patch.pkl'
    cmd = 'serial'
    feature = 'tumor'

    def test_classifies_with_default_args(self, tmp_path):
        path = str(tmp_path)
        level = 1
        cmd = [
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            '-l',
            str(level), input_, '-o', path
        ]
        subprocess.check_call(cmd)
        logger.info('running cmd %s', ' '.join(cmd))
        output_path = os.path.join(path, f'{input_basename}.zarr')
        slide, output = get_input_output(output_path)

        assert output.attrs['filename'] == slide.filename
        assert tuple(output.attrs['resolution']) == slide.dimensions
        print(slide.level_dimensions[level])
        assert output[self.feature].shape == tuple([
            slide.level_dimensions[level][::-1][i] // (128, 128)[i]
            for i in range(2)
        ])
        assert output[self.feature].attrs['extraction_level'] == level
        assert output[self.feature].attrs[
            'level_downsample'] == slide.level_downsamples[level]
        assert output[self.feature].dtype == 'uint8'

    def test_classifies_with_no_round(self, tmp_path):
        path = str(tmp_path)
        level = 1
        cmd = [
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            '-l',
            str(level), '--no-round', input_, '-o', path
        ]
        subprocess.check_call(cmd)
        logger.info('running cmd %s', ' '.join(cmd))
        output_path = os.path.join(path, f'{input_basename}.zarr')
        slide, output = get_input_output(output_path)

        assert output.attrs['filename'] == slide.filename
        assert tuple(output.attrs['resolution']) == slide.dimensions
        assert output[self.feature].shape == tuple([
            slide.level_dimensions[level][::-1][i] // (128, 128)[i]
            for i in range(2)
        ])
        assert output[self.feature].attrs['extraction_level'] == level
        assert output[self.feature].attrs[
            'level_downsample'] == slide.level_downsamples[level]
        assert output[self.feature].dtype == 'float32'
        assert (np.array(output[self.feature]) <= 1).all()


@pytest.mark.parametrize('cmd', ['serial'])
@pytest.mark.parametrize('storage', ['zarr-zip'])
def test_classifies_with_filter(cmd, slide_with_mask, tmp_path,
                                model_all_ones_path, tmpdir, storage):
    path = f'{tmp_path}.{storage}'
    slide = slide_with_mask(np.ones)
    condition = 'mask>2'
    REGISTRY[storage].dump(slide, path)
    REGISTRY[storage].load(path)

    cmd = [
        'classify.py',
        cmd,
        '-f',
        'test',
        '-m',
        model_all_ones_path,
        '-o',
        str(tmpdir),
        '-F',
        f'"{condition}"',
        '--filter-slide',
        path,
        slide.filename,
    ]
    print(' '.join(cmd))
    subprocess.check_call(cmd)


class TestParallelPatchClassifier(TestSerialPatchClassifier):
    cmd = 'parallel'


if __name__ == '__main__':
    unittest.main()
