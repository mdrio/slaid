#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import pickle
import subprocess
import unittest

from slaid.classifiers import TissueFeature
from slaid.commons.ecvl import Slide

DIR = os.path.dirname(os.path.realpath(__file__))
input_ = os.path.join(DIR, 'data/PH10023-1.thumb.tif')
output_pkl = '/tmp/output.pkl'
output_json = '/tmp/output.json'
slide = Slide(input_)


class ExtractTissueTest:
    model = None

    def test_extract_tissue_default_pkl(self):
        subprocess.check_call(
            ['extract_tissue.py', '-m', self.model, input_, output_pkl])
        with open(output_pkl, 'rb') as f:
            data = pickle.load(f)

        self.assertTrue('filename' in data)
        self.assertTrue('extraction_level' in data)
        self.assertTrue('patch_size' in data)
        self.assertTrue('features' in data)
        self.assertTrue(TissueFeature.TISSUE_MASK in data['features'])
        self.assertEqual(data['filename'], input_)
        self.assertEqual(data['patch_size'], (256, 256))
        self.assertEqual(data['extraction_level'], 2)

    def test_extract_tissue_default_json(self):
        subprocess.check_call(
            ['extract_tissue.py', '-m', self.model, input_, output_json])
        with open(output_json, 'r') as f:
            data = json.load(f)
            self.assertTrue('filename' in data)
            self.assertTrue('extraction_level' in data)
            self.assertTrue('patch_size' in data)
            self.assertTrue('features' in data)

    def test_extract_tissue_custom(self):
        extr_level = 1
        cmd = f'extract_tissue.py -m {self.model} -l {extr_level} --no-mask -t 0.7 -T 0.09  {input_} {output_pkl}'
        subprocess.check_call(cmd.split())
        with open(output_pkl, 'rb') as f:
            data = pickle.load(f)
            self.assertTrue('filename' in data)
            self.assertTrue('extraction_level' in data)
            self.assertTrue('patch_size' in data)
            self.assertTrue('features' in data)
            self.assertTrue(TissueFeature.TISSUE_MASK not in data['features'])

            self.assertEqual(data['filename'], input_)
            self.assertEqual(data['patch_size'], (256, 256))
            self.assertEqual(data['extraction_level'], extr_level)

    def test_get_tissue_mask_default_value(self):
        subprocess.check_call([
            'extract_tissue.py', '-m', self.model, '--only-mask', input_,
            output_pkl
        ])
        with open(output_pkl, 'rb') as f:
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
        cmd = f'extract_tissue.py --only-mask -m {self.model} -l {extr_level} -t 0.7 -T 0.09 {input_} {output_pkl}'
        subprocess.check_call(cmd.split())
        slide = Slide(input_, extraction_level=extr_level)
        with open(output_pkl, 'rb') as f:
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
    model = '../slaid/models/extract_tissue_LSVM-1.0.pickle'


class EddlExtractTissueTest(ExtractTissueTest, unittest.TestCase):
    model = '../slaid/models/extract_tissue_eddl-1.0.0.bin'


if __name__ == '__main__':
    unittest.main()
