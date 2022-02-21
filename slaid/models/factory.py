import os

from slaid.models import Factory as BaseFactory, Model
from slaid.models.commons import Factory as CommonFactory
from slaid.models.eddl import Factory as EddlFactory


class Factory(BaseFactory):

    def __init__(self, filename, **kwargs):
        super().__init__(filename)
        self._kwargs = kwargs

    def get_model(self) -> Model:
        _ext_mapping = {
            'pickle': self._get_common_factory,
            'pkl': self._get_common_factory,
            'bin': self._get_eddl_factory,
            'onnx': self._get_onnx_factory
        }
        ext = os.path.splitext(self._filename)[1][1:]
        return _ext_mapping[ext](**self._kwargs).get_model()

    def _get_common_factory(self) -> Model:
        return CommonFactory(self._filename)

    def _get_eddl_factory(self, **kwargs) -> Model:
        return EddlFactory(self._filename, **kwargs)

    def _get_onnx_factory(
            self,
            gpu,
            backend_cls: str = 'slaid.models.eddl.TumorModel') -> Model:
        ...
