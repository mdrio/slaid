#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import subprocess
import unittest

import zarr

from slaid.commons.ecvl import Slide
from slaid.renderers import to_zarr

DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = '/tmp/test-slaid'
input_ = os.path.join(DIR, 'data/PH10023-1.thumb.tif')
input_basename_no_ext = 'PH10023-1.thumb'
slide = Slide(input_)


class BaseTest:
    model = None
    cmd = ''
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
        self.assertEqual(output.attrs['slide'], slide.filename)
        self.assertEqual(output[self.feature].shape,
                         slide.level_dimensions[level][::-1])
        self.assertEqual(output[self.feature].attrs['level'], level)
        self.assertEqual(output[self.feature].attrs['downsample'],
                         slide.level_downsamples[level])

    def test_classifies_with_default_args(self):
        subprocess.check_call([
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            input_, OUTPUT_DIR
        ])
        output_path = os.path.join(OUTPUT_DIR, f'{input_basename_no_ext}.zarr')
        slide, output = self._get_input_output(output_path)

        self._test_output(output, slide, 2)

    def test_classifies_zarr_input(self):
        slide = Slide(input_)
        zarr_path = '/tmp/test-slaid/slide.zarr'
        to_zarr(slide, zarr_path)

        subprocess.check_call([
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            zarr_path, OUTPUT_DIR
        ])
        output_path = os.path.join(OUTPUT_DIR, f'{input_basename_no_ext}.zarr')

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
        output_path = os.path.join(OUTPUT_DIR, f'{input_basename_no_ext}.zarr')
        slide, output = self._get_input_output(output_path)

        self._test_output(output, slide, extr_level)

    def test_overwrites_existing_classification_output(self):
        output = os.path.join(OUTPUT_DIR, f'{input_basename_no_ext}.zarr')
        print(output)
        os.makedirs(output)
        subprocess.check_call([
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            '--overwrite', input_, OUTPUT_DIR
        ])
        stats = os.stat(output)
        self.assertTrue(stats.st_size > 0)
        output_path = os.path.join(OUTPUT_DIR, f'{input_basename_no_ext}.zarr')
        slide, output = self._get_input_output(output_path)

        self._test_output(output, slide, 2)

    def test_raises_error_output_already_exists(self):
        output = os.path.join(OUTPUT_DIR, f'{input_basename_no_ext}.zarr')
        os.makedirs(output)
        subprocess.check_call([
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            '--overwrite', input_, OUTPUT_DIR
        ])

        with self.assertRaises(subprocess.CalledProcessError):
            subprocess.check_call([
                'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
                input_, OUTPUT_DIR
            ])


class TestSerialEddlClassifier(BaseTest, unittest.TestCase):
    model = 'slaid/resources/models/extract_tissue_eddl-1.0.0.bin'
    cmd = 'serial'


class TestParallelEddlClassifier(BaseTest, unittest.TestCase):
    model = 'slaid/resources/models/extract_tissue_eddl-1.0.0.bin'
    cmd = 'parallel'


if __name__ == '__main__':
    unittest.main()
