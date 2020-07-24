from abc import ABC, abstractmethod, abstractproperty
from typing import List, Callable, Any
from commons import Slide
import classifiers


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

    @property
    def next(self):
        return self._next

    @abstractmethod
    def run(self, data_in: Stream):
        pass

    def _post_run(self, output):
        return self.next.run(output) if self.next else output

    def __or__(self, other):
        self._next = other
        return self

    def __add__(self, other):
        return MultiStep([self, other])


class Input(Step):
    def __init__(self, stream: Stream):
        self.stream = stream

    def run(self):
        return self.next.run(self.stream)


class BasicStep(Step):
    def __init__(self, run_method: Callable, *run_args):
        super().__init__()
        self.run_method = run_method
        self.run_args = run_args

    def run(self, data_in: Stream):
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
    return Input(Slide(filename))


def classify(classifier_cls: str, *args):
    return BasicStep(
        getattr(classifiers, classifier_cls).create(*args).classify)


#  slide(slide_filename) | classify('tissue_detection', model) | filter_('tissue' > 0.8) |karolinska(mask_filename) | tiff_render(filename) + json_render(filename)
#  pipeline = slide(
#      '/home/mauro/projects/slide_classifier/tests/integration/input.tiff'
#  ) | classify(
#      'KarolinskaTrueValueClassifier',
#      '/home/mauro/projects/slide_classifier/tests/integration/input.tiff')
#
#  pipeline.run()
