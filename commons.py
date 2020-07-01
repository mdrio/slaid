from openslide import OpenSlide
from collections import namedtuple

Size = namedtuple('Size', ['x', 'y'])
Coordinates = namedtuple('Coordinates', ['x', 'y'])


class Slide(OpenSlide):
    def iterate_by_patch(self, patch_size: Size = None):
        patch_size = patch_size if patch_size else Size(256, 256)
        for x in range(0, self.dimensions[0], patch_size.x):
            for y in range(0, self.dimensions[1], patch_size.y):
                yield Patch(self, Coordinates(x, y), patch_size)


class Patch:
    def __init__(self, slide: Slide, coordinates: Coordinates, size: Size):
        self.slide = slide
        self.x = coordinates.x
        self.y = coordinates.y
        self.size = size

    def __str__(self):
        return (f'slide: {self.slide}, x: {self.x}, '
                f'y: {self.y}, size: {self.size}')
