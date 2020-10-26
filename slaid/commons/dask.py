import logging

import dask
import dask.bag as db
from dask.delayed import delayed
from dask.distributed import Client
from shapely.ops import cascaded_union

from slaid.commons import Mask as BaseMask
from slaid.commons import Polygon

logger = logging.getLogger()


def init_client(*args, **kwargs):
    logger.debug('init dask client with %s, %s', args, kwargs)
    return Client(*args, **kwargs)
    #  import dask
    #  dask.config.set(scheduler='synchronous')


class Mask(BaseMask):
    @delayed
    def _collect_polygons_from_batch(self, batch_idx, batch_size, threshold):
        return super()._collect_polygons_from_batch(batch_idx, batch_size,
                                                    threshold)

    @staticmethod
    def _cascaded_union(geom1, geom2):
        return cascaded_union([geom1, geom2])

    def _get_merged_polygons(self, polygons):
        bag = db.from_delayed(polygons)
        multipolygon = bag.fold(self._cascaded_union).compute()
        bag = db.from_sequence(list(multipolygon))
        return bag.map(lambda p: (list(p.exterior.coords))).map(
            Polygon).compute()
