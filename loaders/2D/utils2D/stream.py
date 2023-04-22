"""
Unwrapping/splitting/batching data samples in the WDS loader.
"""
import torch


def split_masks(mask_index, loader):
    """
    Split the channels in the mask into two separate masks.
    """
    for loader_tuple in loader:
        mask = loader_tuple[mask_index]
        part_mask, mat_mask = torch.split(mask, 1, dim=1)
        yield loader_tuple[:mask_index] + [part_mask.squeeze(1), mat_mask.squeeze(1)] + loader_tuple[mask_index+1:]


def unwrap_cam(cam_tensor):
    """
    Unwrap camera parameters from the loader.
    """
    return cam_tensor[0]
