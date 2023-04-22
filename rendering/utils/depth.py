"""
Writing and manipulating 2D depth images.
"""
import cv2
import numpy as np


def read_exr(filename):
    """
    Read OpenEXR file and convert it to a numpy array.
    """
    depth = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    depth = depth[:, :, 0].astype(np.float32)
    return depth
