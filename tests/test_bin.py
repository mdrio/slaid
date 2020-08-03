#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import subprocess
import unittest

from slaid.commons import Slide

DIR = os.path.dirname(os.path.realpath(__file__))
input_ = os.path.join(DIR, 'data/PH10023-1.svs')
output = '/tmp/output.pkl'
slide = Slide(input_)


class GetTissueMaskTest(unittest.TestCase):
    def test_get_tissue_mask(self):
        subprocess.check_call(['get_tissue_mask.py', input_, output])
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


class ExtractTissueTest(unittest.TestCase):
    def test_extract_tissue(self):
        subprocess.check_call(['extract_tissue.py', input_, output])
        with open(output, 'rb') as f:
            data = pickle.load(f)
            self.assertTrue('filename' in data)
            self.assertTrue('extraction_level' in data)
            self.assertTrue('patch_size' in data)
            self.assertTrue('features' in data)

            self.assertEqual(data['filename'], input_)
            self.assertEqual(data['patch_size'], (256, 256))
            self.assertEqual(data['extraction_level'], 2)


if __name__ == '__main__':
    unittest.main()
