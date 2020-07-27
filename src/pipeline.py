from abc import ABC, abstractmethod
from typing import List, Callable, Any
from enum import Enum
from commons import Slide
import classifiers
from renderers import VectorialRenderer, BasicFeatureTIFFRenderer


class Stream(ABC):
    def __init__(self, data: Any):
        self._data = data

    @staticmethod
    def merge_streams(*streams):
        raise NotImplementedError()

    @abstractmethod
    def merge(self, other):
        pass

    @property
    def data(self):
        return self._data


class SlideStream(Stream):
    def merge(self, other):
        raise NotImplementedError()


class Step(ABC):
    def __init__(self):
        self._next = None
        self._root = None

    @property
    def next(self):
        return self._next

    @property
    def root(self):
        return self._root

    @abstractmethod
    def run(self, data_in: Stream):
        pass

    def _post_run(self, output):
        return self.next.run(output) if self.next else output

    def __or__(self, other):
        self._next = other
        other._root = self.root
        return other

    def __add__(self, other):
        return MultiStep([self, other])


class Input(Step):
    def __init__(self, stream: Stream):
        super().__init__()
        self.stream = stream
        self._root = self

    def run(self):
        return self.next.run(self.stream) if self.next else None


class BasicStep(Step):
    def __init__(self, run_method: Callable, *run_args):
        super().__init__()
        self.run_method = run_method
        self.run_args = run_args

    def run(self, data_in: Stream):
        print(self.run_method)
        print(data_in)
        output = self.run_method(data_in, *self.run_args)
        return self._post_run(output)


class MultiStep(Step):
    def __init__(self, substeps: List[Step]):
        super().__init__()
        self.substeps = substeps

    def run(self, data_in=None):
        outputs = [o for o in self.substeps.run(data_in)]
        output = Stream.merge(outputs)
        return self._post_run(output)

    def __add__(self, other):
        return MultiStep(self.substeps + [other])


def slide(filename: str):
    slide = Slide(filename)
    global _
    _ = slide.patches
    return Input(slide)


def classify(classifier_cls: str, *args):
    try:
        classifier_cls = getattr(classifiers, classifier_cls)
    except AttributeError:
        try:
            classifier_cls = getattr(classifiers,
                                     classifier_cls + 'Classifier')
        except AttributeError:
            raise RuntimeError(f'Classifier {classifier_cls} does not exist')
    return BasicStep(classifier_cls.create(*args).classify)


class RenderOutput(Enum):
    VECT = 'vect'
    TIFF = 'tiff'


_renderers = {
    RenderOutput.VECT: VectorialRenderer,
    RenderOutput.TIFF: BasicFeatureTIFFRenderer
}


def get_renderer(render_output: RenderOutput):
    return _renderers[render_output]()


def render(output: str, filename: str):
    return BasicStep(get_renderer(RenderOutput(output)).render, filename)


if __name__ == '__main__':
    import sys
    pipeline = eval(sys.argv[1])
    pipeline.root.run()
