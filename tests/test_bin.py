#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import shutil
import subprocess
import unittest

import pytest
import zarr

from slaid.commons.ecvl import Slide
from slaid.renderers import from_tiledb, to_zarr

DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = '/tmp/test-slaid'
input_ = os.path.join(DIR, 'data/PH10023-1.thumb.tif')
input_basename = os.path.basename(input_)
input_basename_no_ext = 'PH10023-1.thumb'
slide = Slide(input_)
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)


class TestSerialEddlClassifier:
    model = 'slaid/resources/models/extract_tissue_eddl-1.0.0.bin'
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

    def _get_input_output(self, output):
        slide = Slide(input_)
        zarr_group = zarr.open_group(output)
        return slide, zarr_group

    def _test_output(self, output, slide, level):
        assert output.attrs['slide'] == slide.filename
        assert output[
            self.feature].shape == slide.level_dimensions[level][::-1]
        assert output[self.feature].attrs['level'] == level
        assert output[
            self.feature].attrs['downsample'] == slide.level_downsamples[level]

    def test_classifies_with_default_args(self, tmp_path):
        path = str(tmp_path)
        cmd = [
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            input_, path
        ]
        subprocess.check_call(cmd)
        logger.info('running cmd %s', ' '.join(cmd))
        output_path = os.path.join(path, f'{input_basename}.zarr')
        slide, output = self._get_input_output(output_path)

        self._test_output(output, slide, 2)

    def test_classifies_with_tiledb_as_output(self, tmp_path):
        level = 1
        subprocess.check_call([
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            input_, '-w', 'tiledb', '-l',
            str(level),
            str(tmp_path)
        ])
        output_path = os.path.join(str(tmp_path), f'{input_basename}.tiledb')
        logger.info('checking output_path %s', output_path)
        output = from_tiledb(output_path)

        assert os.path.basename(output.filename) == os.path.basename(
            slide.filename)
        assert output.masks[
            self.feature].array.shape == slide.level_dimensions[level][::-1]
        assert output.masks[self.feature].extraction_level == level
        assert output.masks[
            self.feature].level_downsample == slide.level_downsamples[level]

    def test_classifies_zarr_input(self):
        slide = Slide(input_)
        zarr_path = '/tmp/test-slaid/slide.zarr'
        to_zarr(slide, zarr_path)
        print(os.listdir(zarr_path))
        assert len(os.listdir(zarr_path)) > 0
        print(os)
        print(slide.level_dimensions)

        cmd = [
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            zarr_path, OUTPUT_DIR
        ]
        logger.info('cmd %s', ' '.join(cmd))
        subprocess.check_call(cmd)
        output_path = os.path.join(OUTPUT_DIR, f'{input_basename}.zarr')

        slide, output = self._get_input_output(output_path)
        print(slide, output_path)
        import zarr
        g = zarr.open_group(output_path)
        print(g.attrs.asdict())
        #  self._test_output(output, slide, 2)

    def test_classifies_custom_args(self):
        extr_level = 1
        cmd = f'classify.py {self.cmd} -m {self.model} -f {self.feature}  -l '\
            f' {extr_level}  -t 0.7  {input_} {OUTPUT_DIR}'
        subprocess.check_call(cmd.split())
        output_path = os.path.join(OUTPUT_DIR, f'{input_basename}.zarr')
        slide, output = self._get_input_output(output_path)

        self._test_output(output, slide, extr_level)

    def test_overwrites_existing_classification_output(self):
        output = os.path.join(OUTPUT_DIR, f'{input_basename}.zarr')
        print(output)
        os.makedirs(output)
        subprocess.check_call([
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            '--overwrite', input_, OUTPUT_DIR
        ])
        stats = os.stat(output)
        assert stats.st_size > 0
        output_path = os.path.join(OUTPUT_DIR, f'{input_basename}.zarr')
        slide, output = self._get_input_output(output_path)

        self._test_output(output, slide, 2)

    def test_raises_error_output_already_exists(self):
        output = os.path.join(OUTPUT_DIR, f'{input_basename}.zarr')
        os.makedirs(output)
        subprocess.check_call([
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            '--overwrite', input_, OUTPUT_DIR
        ])

        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call([
                'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
                input_, OUTPUT_DIR
            ])


class TestParallelEddlClassifier(TestSerialEddlClassifier):
    model = 'slaid/resources/models/extract_tissue_eddl-1.0.0.bin'
    cmd = 'parallel'


if __name__ == '__main__':
    unittest.main()
