"""
3DCoMPaT 2D Dataloader Demo.
"""
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T


def make_transform(mode):
    if mode == "train":
        return T.Compose([])
    elif mode == "valid":
        return T.Compose([])


def show_tensors(ims, size, labels=None, font_size=12):
    """
    Display an image in the notebook.
    """
    n_ims = len(ims)
    f, axarr = plt.subplots(1, n_ims)
    f.set_size_inches(size, size * n_ims)
    # set font size and name
    plt.rcParams.update({"font.size": font_size})
    plt.rcParams.update({"font.family": "sans-serif"})
    # set figure title
    if n_ims == 1:
        axarr = [axarr]
    for k, (im, ax) in enumerate(zip(ims, axarr)):
        ax.axis("off")
        im = np.array(torch.squeeze(im).permute(1, 2, 0))
        im = im / np.amax(im)
        # add label
        if labels is not None:
            ax.set_title(labels[k])
        ax.imshow(im)


def mask_to_colors(mask):
    """
    Visualize a mask tensor using a colormap.
    """
    colormap = cm.get_cmap("viridis")

    # Get all unique values in the mask tensor
    unique_values = torch.unique(mask)
    for i, value in enumerate(unique_values):
        mask[mask == value] = i

    # Normalize the mask values to the range [0, 1]
    normalized_mask = mask.float() / mask.max()
    mask_array = normalized_mask.cpu().numpy()

    # Apply the colormap to the mask array
    colored_mask = colormap(mask_array)
    colored_mask_tensor = torch.from_numpy(colored_mask.transpose(0, 3, 1, 2))

    return colored_mask_tensor.unsqueeze(0)


def depth_to_colors(depth):
    """
    Visualize a depth map using a colormap.
    """
    MAX_DEPTH = 100

    colormap = cm.get_cmap("inferno_r")

    # Clip values higher than MAX_DEPTH
    depth[depth > MAX_DEPTH] = np.nan

    # Normalize values
    depth = (depth - np.nanmin(depth)) / (np.nanmax(depth) - np.nanmin(depth))
    depth = depth.cpu().numpy()

    # Replace nans with 1
    depth[np.isnan(depth)] = 1

    # Apply the colormap to the mask array
    colored_depth = colormap(depth)
    colored_depth_tensor = torch.from_numpy(colored_depth.transpose(0, 3, 1, 2))

    return colored_depth_tensor.unsqueeze(0)
