""""
Dataloaders for the preprocessed point clouds from 3DCoMPaT dataset.
"""
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def pc_normalize(pc):
    """
    Center and scale the point cloud.
    """
    pmin = np.min(pc, axis=0)
    pmax = np.max(pc, axis=0)
    pc -= (pmin + pmax) / 2
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc *= 1.0 / scale
    return pc


def load_data(data_dir, partition):
    """
    Pre-load and process the pointcloud data into memory.
    """
    semantic_level = data_dir.split("_")[-2].split("/")[-1]
    h5_name = os.path.join(data_dir, "{}_{}.hdf5".format(partition, semantic_level))
    with h5py.File(h5_name, "r") as f:
        points = np.array(f["points"][:]).astype("float32")
        points_labels = np.array(f["points_labels"][:]).astype("uint16")
        shape_ids = f["shape_id"][:].astype("str")
        shape_labels = np.array(f["shape_label"][:]).astype("uint8")

        normalized_points = np.zeros(points.shape)
        for i in range(points.shape[0]):
            normalized_points[i] = pc_normalize(points[i])

    return normalized_points, points_labels, shape_ids, shape_labels


class CompatLoader3D(Dataset):
    """
    Base class for loading preprocessed 3D point clouds.

    Args:
    ----
        data_root:   Base dataset URL containing data split shards
        split:       One of {train, valid}.
        num_points:  Number of sampled points
        transform:   data transformations
    """

    def __init__(
        self,
        data_root="data/compat",
        split="train",
        num_points=4096,
        transform=None,
    ):
        # train, test, valid
        self.partition = split.lower()
        self.data, self.seg, self.shape_ids, self.label = load_data(
            data_root, self.partition
        )

        self.num_points = num_points
        self.transform = transform

    def __getitem__(self, item):
        idx = np.random.choice(5000, self.num_points, False)
        pointcloud = self.data[item][idx]
        label = self.label[item]
        seg = self.seg[item][idx].astype(np.int32)
        shape_id = self.shape_ids[item]
        pointcloud = torch.from_numpy(pointcloud)
        seg = torch.from_numpy(seg)
        return pointcloud, label, seg, shape_id

    def __len__(self):
        return self.data.shape[0]

    def num_classes(self):
        return np.max(self.label) + 1

    def num_segments(self):
        return np.max(self.seg) + 1

    def get_shape_label(self):
        return self.label


class CompatLoader3DCls(CompatLoader3D):
    """
    Classification data loader using preprocessed 3D point clouds.

    Args:
    ----
        data_root:   Base dataset URL containing data split shards
        split:       One of {train, valid}.
        num_points:  Number of sampled points
        transform:   data transformations
    """

    def __init__(
        self,
        data_root="data/compat",
        split="train",
        num_points=4096,
        transform=None,
    ):
        super().__init__(data_root, split, num_points, transform)

    def __getitem__(self, item):
        idx = np.random.choice(5000, self.num_points, False)
        pointcloud = self.data[item][idx].astype(np.float32)
        label = self.label[item]
        seg = self.seg[item][idx].astype(np.int32)

        pointcloud = torch.from_numpy(pointcloud)
        seg = torch.from_numpy(seg)
        return pointcloud, label
