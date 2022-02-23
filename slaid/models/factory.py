import logging
import os
from urllib.parse import urlparse
from urllib.request import urlretrieve

import slaid.models.eddl as eddl
from slaid.models.base import Factory as BaseFactory
from slaid.models.base import Model
from slaid.models.commons import Factory as CommonFactory

logger = logging.getLogger()

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         '../resources/models')


class Factory(BaseFactory):

    def __init__(self, filename, backend: str = 'eddl', **kwargs):
        filename = retrieve_model(filename)
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

    def _get_onnx_factory(self, gpu=None, cls_name: str = None) -> Model:
        backend_module = self._backends[self.backend]
        factory = getattr(backend_module, 'OnnxFactory')
        return factory(
            self._filename,
            cls_name,
            gpu,
        )
        return factory


def retrieve_model(uri: str) -> str:
    local_path: str

    parsed = urlparse(uri)
    if parsed.scheme in {'http', 'https'}:
        logger.info('retrieving remote model from %s', uri)
        local_path = os.path.join(model_dir, os.path.basename(parsed.path))
        if not os.path.exists(local_path):
            urlretrieve(uri, local_path)
    elif parsed.scheme in {'file', ''}:
        local_path = uri
    else:
        raise UnsupportedScheme(parsed.scheme)

    return local_path


class UnsupportedScheme(Exception):
    ...
