from abc import ABC, abstractmethod
import numpy as np

# TODO: improve this crap
def create_shader_func(shader, **kwargs):
    return lambda x: shader(x, **kwargs)


def checkerboard_shader(coords, size, color_b, color_w):
    sum_coords = np.sum(np.floor(coords / size))
    if sum_coords % 2 == 0:
        return color_b
    return color_w


def constant_shader(coords, color):
    return color
