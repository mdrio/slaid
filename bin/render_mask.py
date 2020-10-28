#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PIL import Image
import pickle


def main(slide_filename, feature, save_to=None):

    with open(slide_filename, 'rb') as f:
        slide = pickle.load(f)
    mask = slide.masks[feature]
    image = Image.fromarray(mask * 255)

    image.show()
    if save_to:
        image.save(save_to)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('slide')
    parser.add_argument('-f', dest='feature', default='tissue')
    parser.add_argument('-s', dest='save_to', default=None)
    args = parser.parse_args()
    main(args.slide, args.feature, args.save_to)
