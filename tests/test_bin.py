#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import subprocess
import unittest

import zarr

from slaid.commons.ecvl import Slide

DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = '/tmp/test-slaid'
input_ = os.path.join(DIR, 'data/PH10023-1.thumb.tif')
input_basename_no_ext = 'PH10023-1.thumb'
slide = Slide(input_)


class ExtractTissueTest:
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
        self.assertEqual(output.attrs['slide'], slide.ID)
        self.assertEqual(output[self.feature].shape,
                         slide.level_dimensions[level][::-1])
        self.assertEqual(output[self.feature].attrs['level'], level)
        self.assertEqual(output[self.feature].attrs['downsample'],
                         slide.level_downsamples[level])

    def test_extract_tissue_default(self):
        subprocess.check_call([
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            input_, OUTPUT_DIR
        ])
        output_path = os.path.join(
            OUTPUT_DIR, f'{input_basename_no_ext}.{self.feature}.zarr')
        slide, output = self._get_input_output(output_path)

        self._test_output(output, slide, 2)

    def test_extract_tissue_custom(self):
        extr_level = 1
        cmd = f'classify.py {self.cmd} -m {self.model} -f {self.feature}  -l '\
            f' {extr_level}  -t 0.7  {input_} {OUTPUT_DIR}'
        subprocess.check_call(cmd.split())
        output_path = os.path.join(
            OUTPUT_DIR, f'{input_basename_no_ext}.{self.feature}.zarr')
        slide, output = self._get_input_output(output_path)

        self._test_output(output, slide, extr_level)

    def test_extract_tissue_overwrite(self):
        output = os.path.join(OUTPUT_DIR,
                              f'{input_basename_no_ext}.{self.feature}.zarr')
        print(output)
        os.makedirs(output)
        subprocess.check_call([
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            '--overwrite', input_, OUTPUT_DIR
        ])
        stats = os.stat(output)
        self.assertTrue(stats.st_size > 0)
        output_path = os.path.join(
            OUTPUT_DIR, f'{input_basename_no_ext}.{self.feature}.zarr')
        slide, output = self._get_input_output(output_path)

        self._test_output(output, slide, 2)

    def test_extract_tissue_no_overwrite(self):
        output = os.path.join(OUTPUT_DIR,
                              f'{input_basename_no_ext}.{self.feature}.zarr')
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

    def test_extract_tissue_skip(self):
        output = os.path.join(OUTPUT_DIR,
                              f'{input_basename_no_ext}.{self.feature}.zarr')
        os.makedirs(output)
        subprocess.check_call(['touch', output])
        subprocess.check_call([
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            '--skip', input_, OUTPUT_DIR
        ])

        slide, output = self._get_input_output(output)
        self.assertEqual(len(list(output.arrays())), 0)


class SerialEddlExtractTissueTest(ExtractTissueTest, unittest.TestCase):
    model = 'slaid/resources/models/extract_tissue_eddl-1.0.0.bin'
    cmd = 'serial'


class ParallelEddlExtractTissueTest(ExtractTissueTest, unittest.TestCase):
    model = 'slaid/resources/models/extract_tissue_eddl-1.0.0.bin'
    cmd = 'parallel'


if __name__ == '__main__':
    unittest.main()
