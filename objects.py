import numpy as np

from gal_renderer.ray_tracing import ray_intersect, intersect_with_scene, get_reflection_vector
from gal_renderer.utils import colorize_pixel, blend_colors
from abc import ABC, abstractmethod


class Triangle:
    # TODO: add intersect function here
    def __init__(self, points, shader):
        self.points = None
        self.v1 = None
        self.v2 = None
        self.area = None
        self.normal = None
        self.update_points(points)
        self.shader = shader

    def update_points(self, new_points):
        self.points = new_points
        self.v1 = self.points[1] - self.points[0]
        self.v2 = self.points[2] - self.points[0]
        cross = np.cross(self.v1, self.v2)
        self.area = np.linalg.norm(cross) / 2
        self.normal = cross / np.linalg.norm(cross)

    def color(self, coords):
        return self.shader(coords)

class Plane:
    def __init__(self, points, shader):
        p0 = points[0]
        p3 = points[1:][np.argmax(np.linalg.norm(points[1:] - p0, axis=1))]
        other_points = [point for point in points[1:] if not np.all(point == p3)]
        points_triangle1 = [p0, other_points[1], other_points[0]]
        points_triangle2 = [p3, other_points[1], other_points[0]]
        self.triangles = [Triangle(points_triangle1, shader),
                          Triangle(points_triangle2, shader)]


class Box:
    def __init__(self, bottom_left, size, shader):
        bottom_face = np.array([bottom_left,
                                bottom_left + np.array([size[0], 0, 0]),
                                bottom_left + np.array([0, size[1], 0]),
                                bottom_left + np.array([size[0], size[1], 0])])
        top_face = bottom_face + np.array([0, 0, size[2]])

        left_face = np.array([bottom_left,
                              bottom_left + np.array([0, 0, size[2]]),
                              bottom_left + np.array([0, size[1], 0]),
                              bottom_left + np.array([0, size[1], size[2]])])
        right_face = left_face + np.array([size[0], 0, 0])

        close_face = np.array([bottom_left,
                               bottom_left + np.array([size[0], 0, 0]),
                               bottom_left + np.array([0, 0, size[2]]),
                               bottom_left + np.array([size[0], 0, size[2]])])
        far_face = close_face + np.array([0, size[1], 0])
        self.triangles = []
        for face in [bottom_face, top_face, left_face, right_face, close_face, far_face]:
            self.triangles += Plane(face, shader).triangles


class LightBaseClass(ABC):
    @abstractmethod
    def get_light_factor(self, triangle, intersect_coords):
        pass


class PointLight(LightBaseClass):
    def __init__(self, position, power):
        self.position = position
        self.power = power

    def get_light_factor(self, triangle, intersect_coords):
        # TODO: this should move to the object itself!!
        vec_to_light = self.position - intersect_coords
        dist_to_light = np.linalg.norm(vec_to_light) + np.finfo(np.float32).eps
        cos_angle = np.abs((vec_to_light / dist_to_light) @ triangle.normal)
        light_factor = self.power * cos_angle / (np.linalg.norm(dist_to_light) ** 2)
        # if cos_angle > np.cos(np.deg2rad(75)):
        #     light_factor *= 1.2

        return light_factor


class Scene:
    def __init__(self, triangles, light_sources):
        self.triangles = triangles
        self.light_sources = light_sources


class Camera:
    def __init__(self, scene, position, orientation, f, fov, n_reflections=3):
        self.position = position
        self.orientation = orientation / np.linalg.norm(orientation)
        self.f = f
        self.fov = fov
        self.scene = scene


    def render(self, im_size, super_sampling=1, n_reflections=1):
        pixel_size = np.tan(self.fov / 2) / (im_size // 2)  # NOT SURE WHY NOT f*_
        image = np.zeros([im_size, im_size, 3])
        # TODO: supersampling on grid? adaptive supersampling? gaussian weighting
        for i in np.arange(-im_size // 2, im_size // 2):
            for j in np.arange(-im_size // 2, im_size // 2):
                ii, jj = i + im_size // 2, j + im_size // 2
                pixel_dir_unnorm_orig = np.array([self.f, i * pixel_size, j * pixel_size]) - self.position
                colors = []
                for iteration in range(super_sampling):
                    if iteration == 0:
                        pixel_dir_unnorm = pixel_dir_unnorm_orig
                    else:
                        pixel_dir_unnorm = pixel_dir_unnorm_orig + np.array(
                            [0, pixel_size * (np.random.random() - 1 / 2), pixel_size * (np.random.random() - 1 / 2)])
                    pixel_direction = pixel_dir_unnorm / np.linalg.norm(pixel_dir_unnorm)
                    color = self.get_color(pixel_direction, self.f, n_reflections)
                    colors.append(color)
                color = np.mean(np.array(colors), axis=0)
                image[ii, jj] = color
        return image

    def get_color(self, pixel_direction, min_allowed_depth, n_reflections):
        EPS_DIST_FROM_SURFACE = 0.00001
        color = np.array([0, 0, 0])
        hit, chosen_tri, chosen_coords = intersect_with_scene(self.scene.triangles, self.position, pixel_direction, min_allowed_depth)
        if hit:
            light_factor = 0
            reflect_colors = []
            for light in self.scene.light_sources:
                t_to_light = np.linalg.norm(light.position - chosen_coords)
                dir_to_light = (light.position - chosen_coords) / t_to_light
                shadowed, _, _ = intersect_with_scene(self.scene.triangles, chosen_coords, dir_to_light,
                                                      EPS_DIST_FROM_SURFACE, early_stop=True, max_t=t_to_light)
                if not shadowed:
                    light_factor += light.get_light_factor(chosen_tri, chosen_coords)

                    reflect_hit_coords = chosen_coords
                    reflect_hit_tri = chosen_tri
                    reflection_direction = pixel_direction
                    for i in range(n_reflections):
                        reflection_direction = get_reflection_vector(reflect_hit_tri, reflection_direction)
                        reflect_hit, reflect_hit_tri, reflect_hit_coords = intersect_with_scene(self.scene.triangles, reflect_hit_coords,
                                                                              reflection_direction, EPS_DIST_FROM_SURFACE)
                        if reflect_hit:
                            reflect_colors += [colorize_pixel(1, reflect_hit_tri, reflect_hit_coords)]
                        else:
                            break

            color = colorize_pixel(light_factor, chosen_tri, chosen_coords)
            if len(reflect_colors) > 0:
                color = blend_colors([color] + reflect_colors)

        return color
