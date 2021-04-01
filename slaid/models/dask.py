import logging

import numpy as np
from dask.distributed import Actor, Client

from slaid.models import Model

logger = logging.getLogger('slaid.models.dask')


class ActorModel(Model):
    @classmethod
    def create(cls, model_cls, **kwargs):
        try:
            client = Client.current()
        except ValueError as ex:
            logger.error(ex)
            client = Client()
        actor = client.submit(model_cls, actor=True, **kwargs).result()
        return cls(actor)

    def __init__(self, actor: Actor):
        self._actor = actor

    def predict(self, array: np.array) -> np.array:
        return self._actor.predict(array).result()

    @property
    def image_info(self):
        return self._actor.image_info

    @property
    def patch_size(self):
        return self._actor.patch_size
