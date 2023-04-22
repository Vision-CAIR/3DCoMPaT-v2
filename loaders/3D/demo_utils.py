"""
3DCoMPaT 3D Dataloader Demo.
"""
import numpy as np
import utils3D.plot as plt_utils


def plot_from_loader(data_loader, n_pc):
    """
    Plot point clouds from a data loader.
    """
    # Iterating over the loader and sampling the first 5 point clouds
    pointclouds = []
    part_labels = []
    for k, (_, _, pointcloud, point_part_labels) in enumerate(data_loader, start=1):
        pointclouds += [pointcloud]
        part_labels += [point_part_labels]
        if k == n_pc:
            break

    # Displaying the first n_pc point clouds
    return plt_utils.plot_pointclouds(
        pointclouds, part_labels, size=15, cmap="viridis", point_size=8
    )


def rotate_pc(pc):
    """
    Rotate a point cloud to align with the camera.
    """
    pc = np.dot(pc, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    return pc


def aug_pc(p_xyz, p_col, shift=1e-3, exp_steps=1):
    """
    Augment the pointcloud with a small random noise for visualization.
    """

    def aug_step(p_xyz, p_col):
        point_noise = np.random.normal(0, shift, p_xyz.shape) + p_xyz
        # Concatenate the original and the augmented pointcloud
        return np.concatenate([p_xyz, point_noise], axis=0), np.concatenate(
            [p_col, p_col], axis=0
        )

    p_xyz_aug, p_col_aug = p_xyz.copy(), p_col.copy()
    for _ in range(exp_steps):
        p_xyz_aug, p_col_aug = aug_step(p_xyz_aug, p_col_aug)
    return p_xyz_aug, p_col_aug
