#!/usr/bin/env python
# -*- coding: utf-8 -*-

from commons import Slide, get_class
import classifiers
import json
from typing import Any, List, Dict
import abc


class Step(abc.ABC):
    @classmethod
    def from_json(cls, json: Dict):
        return cls(json['name'])

    def __init__(self, name: str = ''):
        self._executed = False
        self.name = name

    @property
    def executed(self):
        return self._executed

    @abc.abstractmethod
    def run(self, *args):
        pass

    @abc.abstractproperty
    def input(self) -> Any:
        pass


class Runner(Step):
    @classmethod
    def from_json(cls, json: Dict):
        steps = []
        for step in json['steps']:
            cls_name = step.pop('class')
            module = step.pop('module')
            steps.append(get_class(cls_name, module).from_json(step))
        return cls(steps, json['name'])

    def __init__(self, steps: List[Step], name: str = ''):
        super().__init__(name)
        self._steps = steps

    @property
    def steps(self):
        return self._steps

    @property
    def input(self):
        return self._steps

    def run(self):
        for i, step in enumerate(self.steps):
            if i == 0:
                output = step.run()
            else:
                output = step.run(output)

        self._executed = True


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
