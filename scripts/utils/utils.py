# utils.py  # noqa: ERA001, INP001
import os
import sys

import numpy as np
import scipy

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa: PTH100, PTH118, PTH120

import voxelmorph_folder.voxelmorph as vxm


def interpole_and_pad(image, target_shape=(1024, 1024)):
    """Interpolate and pad image to target shape."""
    if image.shape[0] > target_shape[0] or image.shape[1] > target_shape[1]:
        dim_factors = [target_shape[0] / image.shape[0], target_shape[1] / image.shape[1]]
        image = scipy.ndimage.interpolation.zoom(image, dim_factors, order=0)
    return vxm.py.utils.pad(image, target_shape)[0]


def create_grid(shape, spacing=20):
    """Create a grid image with lines every `spacing` pixels."""
    grid = np.zeros(shape)
    grid[::spacing, :] = 1
    grid[:, ::spacing] = 1
    return grid


def iou(array1, array2, labels=None, include_zero=False):
    """Computes the IoU overlap between two arrays.

    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute dice on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0))

    ioum = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(np.logical_or(array1 == label, array2 == label))
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        ioum[idx] = top / bottom
    return ioum
