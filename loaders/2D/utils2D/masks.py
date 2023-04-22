"""
Masks-related utility functions.
"""
import numpy as np
import cv2
import torch


def from_24bits_RGB(r, g, b):
    """
    Given 3 PyTorch tensors representing 8-bit channels, concatenate them to
    a 24-bit integer for each pixel, and extract the segment code, coarse
    material code and fine material code for each pixel.
    """
    # Concatenate the 3 channels into a single tensor of shape (height, width, 3)
    # convert r,g,b to NumPy arrays
    rgb = np.stack([r, g, b], axis=-1)

    # Convert the 24-bit RGB matrix to a 32-bit integer matrix
    code = np.zeros(rgb.shape[:2], dtype=np.uint32)
    code |= rgb[..., 0].astype(np.uint32) << 16  # Red channel
    code |= rgb[..., 1].astype(np.uint32) << 8   # Green channel
    code |= rgb[..., 2].astype(np.uint32)        # Blue channel

    # Extract the segment code, coarse material code and fine material code
    # for each pixel using bitwise operations and masking
    seg_mask        = (code >> 15) & 0x1FF
    coarse_mat_mask = (code >> 10) & 0x1F
    fine_mat_mask   = code & 0x7F

    # Convert to int64
    seg_mask        = seg_mask.astype(np.int64)
    coarse_mat_mask = coarse_mat_mask.astype(np.int64)
    fine_mat_mask   = fine_mat_mask.astype(np.int64)

    # Convert back to PyTorch tensors
    seg_mask        = torch.from_numpy(seg_mask)
    coarse_mat_mask = torch.from_numpy(coarse_mat_mask)
    fine_mat_mask   = torch.from_numpy(fine_mat_mask)

    ret_tensor = torch.stack([seg_mask, coarse_mat_mask, fine_mat_mask], dim=-1)

    return ret_tensor.permute(2, 0, 1)


def mask_decode(custom_transform, filter=None):
    """
    Extract a segmask, coarse material mask and fine material mask
    from the channels of the torch mask.
    Compose with a custom transform.

    filter:  List of indices for the masks to extract.
                -- 0 = part segmentation mask,
                -- 1 = coarse material mask,
                -- 2 = fine material mask

    Note that the fine material code is only provided as
    a subhierarchy to the coarse material code.
    """
    def transform(mask):
        # Load image from bytes with cv2
        mask = cv2.imdecode(np.frombuffer(mask, np.uint8), cv2.IMREAD_UNCHANGED)
        mask = np.array(mask).astype(np.uint16)
        all_masks = from_24bits_RGB(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])

        # Concate the masks into a single tensor
        if filter is not None:
            # Apply the transform only to the selected mask
            return custom_transform(all_masks[filter])
        return custom_transform(all_masks)
    return transform


def mask_decode_partial(mask_transform, filter, mask):
    """
    Extract a segmask, coarse material mask and fine material mask
    from the channels of the torch mask.
    Compose with a custom transform.

    filter:  List of indices for the masks to extract.
                -- 0 = part segmentation mask,
                -- 1 = coarse material mask,
                -- 2 = fine material mask

    Note that the fine material code is only provided as
    a subhierarchy to the coarse material code.
    """
    # Load image from bytes with cv2
    mask = cv2.imdecode(np.frombuffer(mask, np.uint8), cv2.IMREAD_UNCHANGED)
    mask = np.array(mask).astype(np.uint16)
    all_masks = from_24bits_RGB(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])

    # Concate the masks into a single tensor
    if filter is not None:
        # Apply the transform only to the selected mask
        return mask_transform(all_masks[filter])
    return mask_transform(all_masks)
