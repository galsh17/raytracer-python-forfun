from time import time

from matplotlib import pyplot as plt
import numpy as np
from gal_renderer.obj_parser import parse_obj
from gal_renderer.objects import PointLight, Scene, Camera, Box, Triangle
from gal_renderer.shaders import create_shader_func, constant_shader, checkerboard_shader
from gal_renderer.utils import shape_debugger, translate, rotate


def main():
    # triangles = [Triangle(np.array([[1,-1,-1], [1.2,1,1], [2,1,0]]), color=[244,20,240]),
    # Triangle(np.array([[9,-1,-1], [9,1,1], [9,1,0]]), color=[0,20,240])]
    #Plane(np.array([[3,-1, -1], [3, -1, 1], [3, 1, -1], [3, 1, 1]]), color=[244,20,240]).triangles

    # triangles = Box([10, 10, 3], [4, 8, 4], [97, 200, 45]).triangles +\
    #             Box([8, -5, 5], [1, 1, 1], [150, 0, 86]).triangles +\
    #             Box([3, 4, 4], [1, 1, 1], [200, 0, 200]).triangles +\
    #             Box([3, -3, -2], [1, 1, 1], [124, 80, 100]).triangles
    W = 15
    # triangles += Box([-W, -W, -W], [2*W, 2*W, 2*W], [255,255,0]).triangles
    shader = create_shader_func(constant_shader, color=np.array([255, 255, 255]))
    outer_box_shader = create_shader_func(checkerboard_shader, size=2.5, color_w=np.array([255,255,0]), color_b=np.array([0,255,255]))
    triangles = parse_obj(r'C:\Users\galsh\PycharmProjects\pythonProject\humanoid_tri.obj', outer_box_shader)
    # triangles = rotate(triangles, [d2r(90), 0, 0])
    triangles = translate(triangles, np.array([6, -1, -10]))

    # triangles = rotate(triangles, np.array([np.deg2rad(90), np.deg2rad(90), 0]))
    triangles += Box([-W, -W, -W], [2 * W, 2 * W, 2 * W], shader).triangles
    lights = [PointLight(np.array([2, 3, 7]), 12 ** 2), PointLight(np.array([2, 3, -7]), 7**2)]
    # shape_debugger(triangles, random_color=True)
    scene = Scene(triangles, lights)
    t0 = time()
    cam = Camera(scene=scene, position=np.array([0, 0, 0]), orientation=np.array([1, 0, 0]), f=1, fov=np.deg2rad(110))
    img = cam.render(300, super_sampling=1, n_reflections=2).astype(np.uint8)
    print(f'Render time is: {time() - t0}')
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
