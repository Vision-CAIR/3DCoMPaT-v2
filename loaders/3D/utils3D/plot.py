"""
Utility functions to plot pointclouds with matplotlib.
"""

import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

import numpy as np
import trimesh
from PIL import Image


# Hardcoding the number of parts
# for the demo visualizations
COARSE_PARTS = 43
FINE_PARTS   = 275

COARSE_RGB_RANGE = np.linspace(0, 250, COARSE_PARTS, dtype=int)
FINE_RGB_RANGE   = np.linspace(0, 250, FINE_PARTS, dtype=int)

np.random.seed(0)
np.random.shuffle(COARSE_RGB_RANGE)
np.random.shuffle(FINE_RGB_RANGE)
# ==========================


def label_to_RGB(label_mat, rgb_range):
    """
    Remapping integer labels to RGB colors.
    """
    def get_map(map):
        def f(x):
            return map[x]
        return np.vectorize(f)

    col_map = {}
    col_list = np.unique(label_mat)

    for k, col in enumerate(col_list):
        col_map[col] = rgb_range[k]

    return get_map(col_map)(label_mat)


def crop_plot(margin=0):
    """
    Cropping a 3D Matplotlib projected plot.
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    im = Image.open(buf)

    # Cropping plot
    I = np.asarray(im)
    non_white = np.argwhere(I[:,:,1] != 255)
    y_max, x_max = np.max(non_white, axis=0)
    y_min, x_min = np.min(non_white, axis=0)
    im = im.crop((x_min-margin, y_min-margin, x_max+margin, y_max+margin))

    return im


def concat_figs(ims):
    """
    Concatenate resulting figures.
    """
    widths, heights = zip(*(i.size for i in ims))

    total_width = sum(widths)
    max_height  = max(heights)

    new_im = Image.new('RGBA', (total_width, max_height), color='white')
    x_offset = 0
    for im in ims:
        y_offset = int((max_height - im.size[1])/2)
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]
    return new_im


def get_ratio(p):
    """
    Get the width/depth ratio of the considered object. 
    """
    x_min, y_min, z_min = np.min(p, axis=0)
    x_max, y_max, z_max = np.max(p, axis=0)

    y_range = (y_max - y_min)
    x_ratio = (x_max - x_min)/y_range
    z_ratio = (z_max - z_min)/y_range

    return x_ratio, z_ratio


def radial_sort(p_xyz, p_list):
    """
    Sorts a set of points radially around an
    axis specified by origin and normal vector.
    """
    origin, normal = (0, 0, 0), (1e8, 1e8, 1e8)

    axis0 = [normal[0], normal[2], -normal[1]]
    axis1 = np.cross(normal, axis0)
    vectors = p_xyz - origin

    angles = np.arctan2(np.dot(vectors, axis0),
                        np.dot(vectors, axis1))
    o = angles.argsort()[::-1]

    return [p_xyz[o]] + [p[o] for p in p_list]


def plot_pointclouds(pointclouds,
                     point_labels=None,
                     colors=None,
                     size=8,
                     cmap="viridis",
                     point_size=2,
                     semantic_level="coarse"):
    """
    Displaying point clouds in the notebook.
    """
    global COARSE_RGB_RANGE, FINE_RGB_RANGE
    rgb_range = COARSE_RGB_RANGE if semantic_level == "coarse" else FINE_RGB_RANGE

    def set_ticks(d, ax, p):
        fn = {0: (ax.set_xticks, ax.set_xticklabels),
                    1: (ax.set_yticks, ax.set_yticklabels),
                    2: (ax.set_zticks, ax.set_zticklabels)}
        t = [p[:,d].min(), p[:,d].max()]
        l = [round(v, 2) for v in t]
        fn[d][0](t, minor=False)
        fn[d][1](l, minor=False, fontsize=8)

    def make_fig(k, p, size, color=None, use_cmap=False):
        _ = plt.figure(figsize=(size, size))
        ax = plt.axes(projection='3d')

        # Scaling 3D axes
        x_d, z_d = get_ratio(p)
        x_d = min(x_d, 3.)
        z_d = min(z_d, 3.)

        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax),
                                     np.diag([x_d, 1., z_d, 1.]))

        if color is None:
            ax.scatter(p[:,0], p[:,1], p[:,2],
                       s=point_size, alpha=0.8)
        else:
            if use_cmap:
                ax.scatter(p[:,0], p[:,1], p[:,2],
                           s=point_size, c=color, cmap=cmap, alpha=0.8)
            else:
                ax.scatter(p[:,0], p[:,1], p[:,2],
                           s=point_size, c=color, alpha=0.8)
        ax.dist = 30

        # Setting labels
        ax.set_xlabel('$X$',
                      fontsize=15, fontstyle='normal',
                      fontfamily='monospace', fontweight='bold', color='red')
        ax.set_ylabel('$Y$',
                      fontsize=15, fontstyle='normal',
                      fontfamily='monospace', fontweight='bold', color='green')
        ax.set_zlabel('$Z$',
                      fontsize=15, fontstyle='normal',
                      fontfamily='monospace', fontweight='bold', color='blue')

        # Configuring ticks
        for k in range(3): set_ticks(k, ax, p)
        return crop_plot(margin=20)

    # Generating a subfigure for every shape
    all_ims = []
    for k, p in enumerate(pointclouds):
        if point_labels is None and colors is None:
            all_ims += [make_fig(k, p, size)]
        else:
            use_cmap = False
            if colors is not None:
                clr = colors[k][:,:3]/ 255.0
            elif point_labels is not None:
                clr = label_to_RGB(point_labels[k], rgb_range)
                use_cmap = True

            p, clr = radial_sort(p, [clr])
            all_ims += [make_fig(k, p, size, color=clr, use_cmap=use_cmap)]

    return concat_figs(all_ims)


def show_pointcloud(pointcloud, colors):
    """
    Interactively display a 3D pointcloud through ThreeJS.
    """
    return trimesh.points.PointCloud(pointcloud, colors=colors).scene()
