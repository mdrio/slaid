import os

import slaid.models.eddl as eddl
from slaid.models import Factory as BaseFactory
from slaid.models import Model
from slaid.models.commons import Factory as CommonFactory


class Factory(BaseFactory):

    def __init__(self, filename, backend: str = 'eddl', **kwargs):
        super().__init__(filename)
        self.backend = backend
        self._kwargs = kwargs
        self._backends = {'eddl': eddl}

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
        return eddl.Factory(self._filename, **kwargs)

    def _get_onnx_factory(self, gpu, cls_name: str = None) -> Model:
        backend_module = self._backends[self.backend]
        factory = getattr(backend_module, 'OnnxFactory')
        return factory(
            self._filename,
            cls_name,
            gpu,
        )
        return factory
