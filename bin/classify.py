#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os

import pkg_resources
import tiledb
from clize import run

from slaid.runners import ParallelRunner, SerialRunner

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s '
                    '[%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)


def set_model(func, model):
    def wrapper(*args, **kwargs):
        return func(*args, model=model, **kwargs)

    return wrapper


def set_feature(func, feature):
    def wrapper(*args, **kwargs):
        return func(*args, feature=feature, **kwargs)

    return wrapper


def get_parallel_classifier(model, feature):
    from slaid.classifiers.dask import Classifier
    from slaid.commons.dask import init_client
    init_client()
    return Classifier(model, feature)


def load_config_file(config_file: str, backend: str):
    if config_file is None:
        return
    if backend == 'tiledb':
        config = tiledb.Config.load(config_file)
        tiledb.Ctx(config)
        tiledb.VFS(config)


if __name__ == '__main__':

    runners = {'serial': SerialRunner.run, 'parallel': ParallelRunner.run}
    model = os.environ.get("SLAID_MODEL")
    if model:
        model = pkg_resources.resource_filename('slaid',
                                                f'resources/models/{model}')
        for k, v in runners.items():
            runners[k] = set_model(v, model)

    run(runners)
