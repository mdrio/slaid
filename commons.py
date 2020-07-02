from openslide import OpenSlide
from typing import Tuple


class Slide(OpenSlide):
    def iterate_by_patch(self, patch_size: Tuple[int, int] = None):
        patch_size = patch_size if patch_size else (256, 256)
        for x in range(0, self.dimensions[0], patch_size[0]):
            for y in range(0, self.dimensions[1], patch_size[1]):
                yield Patch(self, (x, y), patch_size)


class Patch:
    def __init__(self, slide: Slide, coordinates: Tuple[int, int],
                 size: Tuple[int, int]):
        self.slide = slide
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size

    def __str__(self):
        return (f'slide: {self.slide}, x: {self.x}, '
                f'y: {self.y}, size: {self.size}')
