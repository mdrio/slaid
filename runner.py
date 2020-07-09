#!/usr/bin/env python
# -*- coding: utf-8 -*-

from commons import Slide
from classifiers import KarolinskaDummyClassifier,\
    BasicFeatureTIFFRenderer, karolinska_text_convert,\
    karolinska_rgb_convert


def classify_mask(in_filename, out_filename):
    mask = Slide(in_filename)

    cl = KarolinskaDummyClassifier(mask)

    features = cl.classify()
    renderer = BasicFeatureTIFFRenderer(karolinska_rgb_convert,
                                        mask.dimensions)
    print('rendering...')
    renderer.render(out_filename, features)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    classify_mask(args.input, args.output)
