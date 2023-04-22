""""
Dataloaders for the preprocessed pointclouds sampled from 3DCoMPaT shapes.
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


def load_data(
    hdf5_path, half_precision=False, normalize_points=False, is_test=False, is_rgb=False
):
    """
    Pre-load and process the pointcloud data into memory.
    """
    if not is_rgb:
        raise NotImplementedError("Only RGB is supported for now.")

    # Loading HDF5 arrays in CPU RAM
    with h5py.File(hdf5_path, "r") as hdf5_f:
        points = np.array(hdf5_f["points"][:]).astype(
            "float32" if not half_precision else "float16"
        )
        shape_ids = np.array(hdf5_f["shape_id"])
        if is_rgb:
            style_ids = np.array(hdf5_f["style_id"])

        if not is_test:
            shape_labels = np.array(hdf5_f["shape_label"])
            points_part_labels = np.array(hdf5_f["points_part_labels"])
            if is_rgb:
                points_mat_labels = np.array(hdf5_f["points_mat_labels"])

        if normalize_points:
            points = np.zeros(points.shape)
            for i in range(points.shape[0]):
                points[i] = pc_normalize(points[i])

    # Alternative: return a list of tensors
    ret_list = [points, shape_ids]
    if is_rgb:
        ret_list.append(style_ids)

    if not is_test:
        ret_list.append(shape_labels)
        ret_list.append(points_part_labels)
        if is_rgb:
            ret_list.append(points_mat_labels)

    return ret_list


class CompatLoader3D_PC(Dataset):
    """
    Base class for loading preprocessed 3D point clouds.

    Args:
    ----
        root_dir:          Root directory containing HDF5 files.
        split:             One of {train, valid}.
        semantic_level:    Semantic level to use for segmentations. One of {fine, medium, coarse}
        num_points:        Number of points to sample.
        transform:         Data transformations.
        half_precision:    Use half precision floats.
        normalize_points:  Normalize point clouds.
        is_rgb:            The HDF5 to load has RGB features.
    """

    def __init__(
        self,
        root_dir,
        split,
        semantic_level,
        num_points=4096,
        transform=None,
        half_precision=False,
        normalize_points=False,
        is_rgb=True,
    ):
        if split not in ["train", "valid", "test"]:
            raise ValueError("Invalid split: {}".format(split))
        file_suffix = "_no_gt" if split == "test" else ""
        hdf5_path = os.path.join(
            root_dir, "%s_%s.hdf5" % (split, semantic_level + file_suffix)
        )
        self.is_test = split == "test"
        self.is_rgb = is_rgb
        self.hdf5_data = load_data(
            hdf5_path,
            half_precision,
            normalize_points=normalize_points,
            is_test=self.is_test,
            is_rgb=is_rgb,
        )
        self.num_points = num_points
        self.points = self.hdf5_data[0]
        self.max_points = self.points.shape[1]
        self.transform = transform
        if self.num_points > self.max_points:
            raise ValueError(
                "Requested to sample more points than available in the dataset."
            )

        # Loading shape_ids, style_ids
        self.shape_ids = self.hdf5_data[1]
        self.style_ids = self.hdf5_data[2]

        # Converting all elements to strings from bytes
        self.shape_ids = [str(x, "utf-8") for x in self.shape_ids]
        self.style_ids = [str(x, "utf-8") for x in self.style_ids]

        model_style_ids = [
            shape_id + "_" + style_id
            for shape_id, style_id in zip(self.shape_ids, self.style_ids)
        ]
        self.model_style_map = {
            model_style_id: i for i, model_style_id in enumerate(model_style_ids)
        }

    def __getitem__(self, item):
        pass

    def __len__(self):
        return self.points.shape[0]

    def get_stylized_shape(self, shape_id, style_id):
        return self.__getitem__(self.model_style_map[shape_id + "_" + str(style_id)])

    def num_classes(self):
        return np.max(self.shape_labels) + 1

    def num_segments(self):
        return np.max(self.seg) + 1

    def get_shape_label(self):
        return self.shape_labels


class StylizedShapeLoader_PC(CompatLoader3D_PC):
    """
    StylizedShapeLoader for loading stylized 3D point clouds.

    Args:
    ----
        ...:    See CompatLoader3D_PC.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        idx = np.random.choice(self.max_points, self.num_points, True)

        points = torch.from_numpy(self.points[item][idx].astype(np.float32))

        # Unwrapping the HDF5 data
        shape_id = self.shape_ids[item]
        style_id = self.style_ids[item]
        shape_label = self.hdf5_data[3][item]

        # Load the part labels
        points_part_labels = self.hdf5_data[4]
        points_part_labels = torch.from_numpy(
            points_part_labels[item][idx].astype(np.int16)
        )

        # Also load the material labels
        points_mat_labels = self.hdf5_data[5]
        points_mat_labels = torch.from_numpy(
            points_mat_labels[item][idx].astype(np.uint8)
        )

        return (
            shape_id,
            style_id,
            shape_label,
            points,
            points_part_labels,
            points_mat_labels,
        )

    def get_stylized_shape(self, shape_id, style_id):
        return super().get_stylized_shape(shape_id, style_id)


class EvalLoader_PC(CompatLoader3D_PC):
    """
    StylizedShapeLoader for loading stylized 3D point clouds.

    Args:
    ----
        ...:    See CompatLoader3D_PC.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, split="test", **kwargs)

    def __getitem__(self, item):
        idx = np.arange(self.num_points)

        points = torch.from_numpy(self.points[item][idx].astype(np.float32))

        # Unwrapping the HDF5 data
        shape_id = self.shape_ids[item]
        style_id = self.style_ids[item]

        return shape_id, style_id, points

    def get_stylized_shape(self, shape_id, style_id):
        return super().get_stylized_shape(shape_id, style_id)
