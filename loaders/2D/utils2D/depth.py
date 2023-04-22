"""
Reading and manipulating 2D depth images.
"""
import cv2
import numpy as np


def depth_decode(depth_transform, depth):
    """
    Extract a depth map in EXR format from a byte buffer.
    """
    # Load image from bytes with cv2
    depth = cv2.imdecode(np.frombuffer(depth, np.uint8),
                            cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    depth = np.array(depth).astype(np.float32)

    return depth_transform(depth)
