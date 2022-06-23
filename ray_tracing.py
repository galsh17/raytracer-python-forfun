import numpy as np
from numba import jit


# def ray_intersect(origin, direction, triangle):
#     #TODO: return reflection vector. what to do if not inv
#
#     beta, gamma, t = np.linalg.solve(
#         np.array([triangle.v1, triangle.v2, direction]).T,
#         np.array(triangle.points[0] - origin)
#     )
#     # print(beta, gamma, t)
#     if t < 0 or beta > 1 or gamma > 1 or beta < 0 or gamma < 0:
#         t = np.inf
#     return t, origin + t * direction if t is not np.inf else np.inf
#
#

@jit(nopython=True)
def numba_ray_intersect(origin, direction, v0v1, v0v2, points):
    pvec = np.cross(direction, v0v2)
    det = np.dot(v0v1, pvec)
    if abs(det) < 0.000000001:
        return origin + direction, -100000000000.0  # np.inf, np.inf
    # TODO: WTF ?
    if det < 0:
        v0v1, v0v2 = v0v2, v0v1
        pvec = np.cross(direction, v0v2)
        det = np.dot(v0v1, pvec)

    invDet = 1.0 / det
    tvec = origin - points[0]
    u = np.dot(tvec, pvec) * invDet
    if u < 0 or u > 1.0:
        return origin + direction, -100000000000.0  # np.inf, np.inf
    qvec = np.cross(tvec, v0v1)
    v = np.dot(direction, qvec) * invDet

    if v < 0 or u + v > 1.0 + 2 * np.finfo(np.float32).eps:
        return origin + direction, -100000000000.0  # np.inf, np.inf
    t = np.dot(v0v2, qvec) * invDet
    intersection_point = origin + direction * t
    return intersection_point, t


def ray_intersect(origin, direction, triangle):
    v1 = np.array(triangle.v1, dtype=np.float64)
    v2 = np.array(triangle.v2, dtype=np.float64)
    points = np.array(triangle.points, dtype=np.float64)
    return numba_ray_intersect(origin, direction, v1, v2, points)


def intersect_with_scene(all_triangles, start_position, direction, min_allowed_t, early_stop=False, max_t=None):
    hit = False
    chosen_coords = None
    chosen_tri = None
    if max_t is not None:
        min_depth = max_t
    else:
        min_depth = np.inf
    for tri in all_triangles:
        coords, t = ray_intersect(start_position, direction, tri)
        if min_depth > t > min_allowed_t:
            min_depth = t
            chosen_tri = tri
            chosen_coords = coords
            hit = True
            if early_stop:
                break
    return hit, chosen_tri, chosen_coords


def get_reflection_vector(triangle, ray_direction):
    correct_normal = triangle.normal
    dot_ans = np.dot(correct_normal, ray_direction)
    if dot_ans > 0:
        correct_normal = -correct_normal
        dot_ans = -dot_ans
    reflection = ray_direction - 2 * dot_ans * correct_normal
    return reflection / np.linalg.norm(reflection)

# def ray_intersect(origin, direction, triangle):
#     v0v1 = triangle.v1
#     v0v2 = triangle.v2
#     pvec = np.cross(direction, v0v2) #r.direction.cross(v0v2)
#
#     det = np.dot(v0v1, pvec)
#     if abs(det) < 0.000000001: # abs?
#         return np.inf, np.inf
#
#     # TODO: WTF ?
#     if det < 0:
#         v0v1 = triangle.v2
#         v0v2 = triangle.v1
#         pvec = np.cross(direction, v0v2)  # r.direction.cross(v0v2)
#         det = np.dot(v0v1, pvec)
#
#     invDet = 1.0 / det
#     tvec = origin - triangle.points[0]
#     u = np.dot(tvec, pvec) * invDet
#
#     if u < 0 or u > 1.0:
#         return np.inf, np.inf
#
#     qvec = np.cross(tvec, v0v1)
#     v = np.dot(direction, qvec) * invDet
#
#     if v < 0 or u + v > 1.0 + 2*np.finfo(np.float32).eps:
#         return np.inf, np.inf
#     t = np.dot(v0v2, qvec) * invDet
#     intersection_point = origin + direction*t
#     return intersection_point, t
#
