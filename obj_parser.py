from gal_renderer.objects import Triangle
import numpy as np


def get_vertices(obj_file):
    return np.array([np.fromstring(l[2:], sep=' ')[:3] for l in obj_file if l.startswith('v')])


def get_triangles(obj_file, vertices, shader):
    faces_idxes = np.array([np.fromstring(l[2:], sep=' ', dtype=int) for l in obj_file if l.startswith('f')])
    return [Triangle(vertices[face_idx-1], shader) for face_idx in faces_idxes]


def parse_obj(path, shader):
    with open(path, 'r') as f:
        lines = f.readlines()
    verts = get_vertices(lines)
    triangles = get_triangles(lines, verts, shader)
    return triangles
