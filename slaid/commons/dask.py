import logging

from dask.distributed import Client

logger = logging.getLogger()


def init_client(*args, **kwargs):
    logger.debug('init dask client with %s, %s', args, kwargs)
    return Client(*args, **kwargs)
    #  import dask
    #  dask.config.set(scheduler='synchronous')
