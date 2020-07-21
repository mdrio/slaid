#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import json
from typing import List, Dict, Any
from commons import Slide, get_class
import classifiers


class Step(abc.ABC):
    @classmethod
    def from_json(cls, json: Dict):
        return cls(json['name'])

    def __init__(self, name: str = '', _input: Any = None):
        self._input = _input
        self.name = name
        self._executed = False

    @property
    def input(self):
        return self._input

    @property
    def executed(self):
        return self._executed

    @abc.abstractmethod
    def run(self, *args):
        pass


class ClassifierStep(Step):
    def __init__(self, classifier: classifiers.Classifier):
        self.classifier = classifier

    def run(self, patch_collection: classifiers.PatchCollection):
        return self.classifier.classify(patch_collection)


class Runner(Step):
    @classmethod
    def from_json(cls, json: Dict):
        steps = []
        for step in json['steps']:
            cls_name = step.pop('class')
            module = step.pop('module')
            steps.append(get_class(cls_name, module).from_json(step))
        return cls(steps, json['name'])

    def __init__(self, steps: List[Step], name: str = '', _input: Any = None):
        super().__init__(name, _input)
        self._steps = steps
        self._input = _input

    @property
    def steps(self):
        return self._steps

    @property
    def input(self):
        return self._input

    def run(self):
        _input = self.input
        for i, step in enumerate(self.steps):
            _input = step.run(_input)
        return _input


def main(classifier_name, in_filename, tiff_filename, json_filename, *args):
    slide = Slide(in_filename)

    cl = getattr(classifiers, classifier_name).create(*args)

    patches = classifiers.PandasPatchCollection(slide, (256, 256))
    features = cl.classify(patches)
    with open(json_filename, 'w') as json_file:
        json.dump(features, json_file, cls=classifiers.JSONEncoder)

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
