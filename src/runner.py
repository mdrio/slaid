#!/usr/bin/env python
# -*- coding: utf-8 -*-

from commons import Slide
import classifiers
import json


def main(classifier_name, in_filename, tiff_filename, json_filename, *args):
    slide = Slide(in_filename)

    cl = getattr(classifiers, classifier_name).create(*args)

    features = cl.classify(slide)
    with open(json_filename, 'w') as json_file:
        json.dump(features, json_file, cls=classifiers.PatchFeatureJsonEncoder)

    renderer = classifiers.BasicFeatureTIFFRenderer(
        classifiers.karolinska_rgb_convert, slide.dimensions)
    print('rendering...')
    renderer.render(tiff_filename, features)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('-t', dest='tiff_filename')
    parser.add_argument('-c',
                        dest='classifier',
                        default='KarolinskaTrueValueClassifier')
    parser.add_argument('classifier_args', nargs='*')
    parser.add_argument('-j', dest='json_filename')

    args = parser.parse_args()

    main(args.classifier, args.input, args.tiff_filename, args.json_filename,
         *args.classifier_args)
