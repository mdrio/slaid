#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import pickle
import subprocess
import unittest

from slaid.commons.ecvl import Slide

DIR = os.path.dirname(os.path.realpath(__file__))
input_ = os.path.join(DIR, 'data/PH10023-1.thumb.tif')
slide = Slide(input_)


class ExtractTissueTest:
    model = None

    def test_extract_tissue_default_pkl(self):
        subprocess.check_call(
            ['classify.py', '-w', 'pkl', '-m', self.model, input_])
        with open(f'{input_}.output.pkl', 'rb') as f:
            data = pickle.load(f)

        self.assertTrue('filename' in data)
        self.assertTrue('extraction_level' in data)
        self.assertTrue('patch_size' in data)
        self.assertTrue('features' in data)
        self.assertTrue('tissue' in set(data['features'].columns))
        self.assertEqual(data['filename'], input_)
        self.assertEqual(data['patch_size'], (256, 256))
        self.assertEqual(data['extraction_level'], 2)

    def test_extract_tissue_only_mask_pkl(self):
        subprocess.check_call([
            'classify.py', '--only-mask', '-w', 'pkl', '-m', self.model, input_
        ])
        with open(f'{input_}.output.pkl', 'rb') as f:
            data = pickle.load(f)

        self.assertTrue('filename' in data)
        self.assertTrue('mask' in data)
        self.assertTrue('dimensions' in data)

        self.assertTrue('extraction_level' in data)
        self.assertEqual(data['extraction_level'], 2)

    def test_extract_tissue_default_json(self):
        subprocess.check_call(
            ['classify.py', '-w', 'json', '-m', self.model, input_])
        with open(f'{input_}.output.json', 'rb') as f:
            data = json.load(f)
            self.assertTrue('filename' in data)
            self.assertTrue('extraction_level' in data)
            self.assertTrue('patch_size' in data)
            self.assertTrue('features' in data)

    def test_extract_tissue_custom(self):
        extr_level = 1
        cmd = f'classify.py -m {self.model} -w pkl -l {extr_level} --no-mask -t 0.7 -T 0.09 {input_}'
        subprocess.check_call(cmd.split())
        with open(f'{input_}.output.pkl', 'rb') as f:
            data = pickle.load(f)
            self.assertTrue('filename' in data)
            self.assertTrue('extraction_level' in data)
            self.assertTrue('patch_size' in data)
            self.assertTrue('features' in data)
            self.assertTrue('tissue' in set(data['features'].columns))

            self.assertEqual(data['filename'], input_)
            self.assertEqual(data['patch_size'], (256, 256))
            self.assertEqual(data['extraction_level'], extr_level)

    def test_get_tissue_mask_default_value(self):
        subprocess.check_call([
            'classify.py', '-m', self.model, '-w', 'pkl', '--only-mask', input_
        ])
        with open(f'{input_}.output.pkl', 'rb') as f:
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
        cmd = f'classify.py --only-mask -m {self.model} -l {extr_level} -w pkl -t 0.7 -T 0.09 {input_}'
        subprocess.check_call(cmd.split())
        slide = Slide(input_, extraction_level=extr_level)
        with open(f'{input_}.output.pkl', 'rb') as f:
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


class SVMExtractTissueTest(ExtractTissueTest, unittest.TestCase):
    model = 'slaid/models/extract_tissue_LSVM-1.0.pickle'


class EddlExtractTissueTest(ExtractTissueTest, unittest.TestCase):
    model = 'slaid/models/extract_tissue_eddl-1.0.0.bin'


if __name__ == '__main__':
    unittest.main()
