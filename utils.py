import numpy as np
from skimage import color as skcolor


def blend_colors(colors):
    # TODO: weighted mean?
    # assumes illumination from first color
    hs_list = []
    n_colors = len(colors)
    weights = [0.6, 0.4]
    for i, color in enumerate(colors):
        hsv = skcolor.rgb2hsv(color / 255)
        if i == 0:
            final_v = hsv[-1]
            first_hs = hsv[:-1]
            continue
        hs_list += [hsv[:-1]]
    if n_colors > 2:
        hs_mean = np.mean(hs_list, axis=0)
    else:
        hs_mean = hsv[:-1]
    lala = np.average([first_hs, hs_mean], axis=0, weights=weights)
    hsv = np.append(lala, final_v)
    return skcolor.hsv2rgb(hsv) * 255


def colorize_pixel(light_factor, triangle, coords):
    hsv = skcolor.rgb2hsv(np.array(triangle.color(coords)) / 255)
    hsv[-1] = min(hsv[-1] * light_factor, 1)
    return skcolor.hsv2rgb(hsv) * 255


def shape_debugger(triangles, random_color=False):
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    for tri in triangles:
        polygon = Poly3DCollection([tri.points])
        if random_color:
            color = np.random.random(3)
        else:
            color = tri.color // 255
        polygon.set_color(color)
        ax.add_collection3d(polygon)
    plt.show()


def translate(triangles, translation):
    for tri in triangles:
        tri.points += translation
    return triangles


def get_rot_mat(rotation):
    def Rx(theta):
        return np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta), np.cos(theta)]])

    def Ry(theta):
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])

    def Rz(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])

    psi, theta, phi = rotation
    R = Rz(psi) @ Ry(theta) @ Rx(phi)
    return R


def rotate(triangles, rotation):
    rotation_matrix = get_rot_mat(rotation)
    return [tri.update_points(tri.points @ rotation_matrix) for tri in triangles]
