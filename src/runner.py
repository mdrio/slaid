#!/usr/bin/env python
# -*- coding: utf-8 -*-

from commons import Slide
import classifiers


def classify_mask(in_filename, out_filename, classifier_name, *args):
    slide = Slide(in_filename)

    cl = getattr(classifiers, classifier_name).create(*args)

    features = cl.classify(slide)
    renderer = classifiers.BasicFeatureTIFFRenderer(
        classifiers.karolinska_rgb_convert, slide.dimensions)
    print('rendering...')
    renderer.render(out_filename, features)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('-c',
                        dest='classifier',
                        default='KarolinskaTrueValueClassifier')
    parser.add_argument('classifier_args', nargs='*')

    args = parser.parse_args()

    classify_mask(args.input, args.output, args.classifier,
                  *args.classifier_args)
