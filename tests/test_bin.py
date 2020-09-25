#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import pickle
import shutil
import subprocess
import unittest
from tempfile import NamedTemporaryFile

from slaid.commons.ecvl import Slide, create_slide

DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = '/tmp/test-slaid'
input_ = os.path.join(DIR, 'data/PH10023-1.thumb.tif')
input_basename_no_ext = 'PH10023-1.thumb'
slide = Slide(input_)


class ExtractTissueTest:
    model = None
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

    def _test_pickled(self, slide_pickled, extraction_level):
        self.assertTrue(isinstance(slide_pickled, Slide))
        self.assertTrue('tissue' in set(slide_pickled.patches.features))
        self.assertEqual(slide_pickled.ID, os.path.basename(input_))
        self.assertEqual(slide_pickled.patches.patch_size, (256, 256))
        self.assertEqual(slide_pickled.patches.extraction_level,
                         extraction_level)
        self.assertTrue(slide_pickled.patches.dataframe.size > 0)
        #  self.assertEqual(
        #      slide_pickled.patches.dataframe.size,
        #      slide_pickled.dimensions_at_extraction_level[0] *
        #      slide_pickled.dimensions_at_extraction_level[1] //
        #      (slide_pickled.patches.patch_size[0] *
        #       slide_pickled.patches.patch_size[1]))

    def test_extract_tissue_default_pkl(self):
        subprocess.check_call([
            'classify.py', '-f', self.feature, '-w', 'pkl', '-m', self.model,
            input_, OUTPUT_DIR
        ])
        output = os.path.join(OUTPUT_DIR,
                              f'{input_basename_no_ext}.{self.feature}.pkl')
        with open(output, 'rb') as f:
            slide_pickled = pickle.load(f)
            self._test_pickled(slide_pickled, 2)

    def test_extract_tissue_only_mask_pkl(self):
        subprocess.check_call([
            'classify.py', '--only-mask', '-f', self.feature, '-w', 'pkl',
            '-m', self.model, input_, OUTPUT_DIR
        ])
        output = os.path.join(OUTPUT_DIR,
                              f'{input_basename_no_ext}.{self.feature}.pkl')
        with open(output, 'rb') as f:
            data = pickle.load(f)

        self.assertTrue('filename' in data)
        self.assertTrue('mask' in data)
        self.assertTrue('dimensions' in data)

        self.assertTrue('extraction_level' in data)
        self.assertEqual(data['extraction_level'], 2)

    def test_extract_tissue_default_json(self):
        subprocess.check_call([
            'classify.py', '-f', self.feature, '-w', 'json', '-m', self.model,
            input_, OUTPUT_DIR
        ])
        output = os.path.join(OUTPUT_DIR,
                              f'{input_basename_no_ext}.{self.feature}.json')
        with open(output, 'rb') as f:
            data = json.load(f)
            self.assertTrue('filename' in data)
            self.assertTrue('extraction_level' in data)
            self.assertTrue('patch_size' in data)
            self.assertTrue('features' in data)

    def test_extract_tissue_custom(self):
        extr_level = 1
        cmd = f'classify.py -m {self.model} -f {self.feature} -w pkl -l {extr_level} --no-mask -t 0.7 -T 0.09 {input_} {OUTPUT_DIR}'
        subprocess.check_call(cmd.split())
        output = os.path.join(OUTPUT_DIR,
                              f'{input_basename_no_ext}.{self.feature}.pkl')
        with open(output, 'rb') as f:
            slide_pickled = pickle.load(f)
            self._test_pickled(slide_pickled, 1)
        os.remove(output)

    def test_input_as_pickle(self):
        slide = create_slide(input_, 2)
        with NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(slide, f)
        extr_level = 1
        cmd = f'classify.py -m {self.model} -f {self.feature} -l {extr_level} {f.name} {OUTPUT_DIR}'
        subprocess.check_call(cmd.split())
        output = os.path.join(
            OUTPUT_DIR,
            f'{os.path.splitext(os.path.basename(f.name))[0]}.{self.feature}.pkl'
        )

        with open(output, 'rb') as f:
            pickle.load(f)
        os.remove(f.name)

    def test_input_as_pickle_with_filter(self):
        slide = create_slide(input_, 2)
        slide.patches.add_feature('feature')
        slide.patches.update_patch((0, 0), features={'feature': 1})
        with NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(slide, f)
        extr_level = 1
        cmd = f'classify.py -m {self.model} -f {self.feature} -l {extr_level} -F feature>=1 {f.name} {OUTPUT_DIR}'
        subprocess.check_call(cmd.split())
        output = os.path.join(
            OUTPUT_DIR,
            f'{os.path.splitext(os.path.basename(f.name))[0]}.{self.feature}.pkl'
        )

        with open(output, 'rb') as f_out:
            slide_pickled = pickle.load(f_out)

        self.assertTrue(slide_pickled.patches.dataframe.shape == (1, 2))
        os.remove(f.name)

    def test_get_tissue_mask_default_value(self):
        subprocess.check_call([
            'classify.py', '-m', self.model, '-w', 'pkl', '-f', self.feature,
            '--only-mask', input_, OUTPUT_DIR
        ])
        output = os.path.join(OUTPUT_DIR,
                              f'{input_basename_no_ext}.{self.feature}.pkl')
        with open(output, 'rb') as f:
            data = pickle.load(f)
        self.assertTrue('filename' in data)
        self.assertTrue('extraction_level' in data)
        self.assertTrue('mask' in data)

        self.assertEqual(data['filename'], input_)
        self.assertEqual(data['extraction_level'], 2)
        self.assertEqual(data['mask'].transpose().shape,
                         slide.dimensions_at_extraction_level)
        self.assertTrue(sum(sum(data['mask'])) > 0)

    def test_get_tissue_mask_custom_value(self):
        extr_level = 1
        cmd = f'classify.py -f {self.feature} --only-mask -m {self.model} -l {extr_level} -w pkl -t 0.7 -T 0.09 {input_} {OUTPUT_DIR}'
        subprocess.check_call(cmd.split())
        slide = Slide(input_, extraction_level=extr_level)
        output = os.path.join(OUTPUT_DIR,
                              f'{input_basename_no_ext}.{self.feature}.pkl')
        with open(output, 'rb') as f:
            data = pickle.load(f)
        self.assertTrue('filename' in data)
        self.assertTrue('extraction_level' in data)
        self.assertTrue('mask' in data)

        self.assertEqual(data['filename'], input_)
        self.assertEqual(data['dimensions'], slide.dimensions)
        self.assertEqual(data['extraction_level'], extr_level)
        self.assertEqual(data['mask'].transpose().shape,
                         slide.dimensions_at_extraction_level)
        self.assertTrue(sum(sum(data['mask'])) > 0)

    def test_extract_tissue_overwrite(self):
        output = os.path.join(OUTPUT_DIR,
                              f'{input_basename_no_ext}.{self.feature}.pkl')
        os.makedirs(OUTPUT_DIR)
        subprocess.check_call(['touch', output])
        subprocess.check_call([
            'classify.py', '-f', self.feature, '-w', 'pkl', '-m', self.model,
            '--overwrite', input_, OUTPUT_DIR
        ])
        stats = os.stat(output)
        self.assertTrue(stats.st_size > 0)

        with open(output, 'rb') as f:
            slide_pickled = pickle.load(f)
            self._test_pickled(slide_pickled, 2)

    def test_extract_tissue_no_overwrite(self):
        output = os.path.join(OUTPUT_DIR,
                              f'{input_basename_no_ext}.{self.feature}.pkl')
        os.makedirs(OUTPUT_DIR)
        subprocess.check_call(['touch', output])
        with self.assertRaises(subprocess.CalledProcessError):

            subprocess.check_call([
                'classify.py', '-f', self.feature, '-w', 'pkl', '-m',
                self.model, input_, OUTPUT_DIR
            ])


def test_extract_tissue_skip(self):
    output = os.path.join(OUTPUT_DIR,
                          f'{input_basename_no_ext}.{self.feature}.pkl')
    os.makedirs(OUTPUT_DIR)
    subprocess.check_call(['touch', output])
    subprocess.check_call([
        'classify.py', '-f', self.feature, '-w', 'pkl', '-m', self.model,
        '--skip', input_, OUTPUT_DIR
    ])
    stats = os.stat(output)
    self.assertEqual(stats.st_size, 0)

    with open(output, 'rb') as f:
        slide_pickled = pickle.load(f)
        self._test_pickled(slide_pickled, 2)


class SVMExtractTissueTest(ExtractTissueTest, unittest.TestCase):
    model = 'slaid/resources/models/extract_tissue_LSVM-1.0.pickle'


class EddlExtractTissueTest(ExtractTissueTest, unittest.TestCase):
    model = 'slaid/resources/models/extract_tissue_eddl-1.0.0.bin'


if __name__ == '__main__':
    unittest.main()
