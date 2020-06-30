from openslide import OpenSlide
from collections import namedtuple


Size = namedtuple('Size', ['x', 'y'])
Coordinates = namedtuple('Coordinates', ['x', 'y'])


class Patch(object):
    def __init__(self, slide: OpenSlide, coordinates: Coordinates, size: Size):
        self.slide = slide
        self.x = coordinates.x
        self.y = coordinates.y
        self.size = size
