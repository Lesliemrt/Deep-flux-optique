# utils.py  # noqa: ERA001, INP001
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import scipy
import tifffile as tiff
from skimage.measure import regionprops

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa: PTH100, PTH118, PTH120

import voxelmorph_folder.voxelmorph as vxm
from raft.core.utils import flow_viz


def normalisation_params(frames_path_list):
    """Return the global minimum and maximum pixel values across all frame's paths in the video."""
    global_min = np.min(tiff.imread(str(frames_path_list[0])))
    global_max = np.max(tiff.imread(str(frames_path_list[0])))

    for i in range(1, len(frames_path_list)):
        current_min = np.min(tiff.imread(str(frames_path_list[i])))
        current_max = np.max(tiff.imread(str(frames_path_list[i])))

        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)

    return global_min, global_max


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
        ioum[idx] = top / bottom if bottom > 0 else 0.0
    return ioum


def relabel_cells(array, with_labels):
    """Relabel an segmentation mask with labels from another one."""
    relabeled = np.zeros_like(array)
    for region in regionprops(array):
        if region.label == 0:
            continue
        # Get the most common label in the region from with_labels
        labels_in_region = with_labels[region.coords[:, 0], region.coords[:, 1]]
        label_count = Counter(labels_in_region)
        if max(label_count.keys()) == 0:
            most_common_label = 0
        else:
            label_count.pop(0, None)
            most_common_label = label_count.most_common(1)[0][0]
        relabeled[region.coords[:, 0], region.coords[:, 1]] = most_common_label
    return relabeled


def visualize_labels(mask, ax=None):
    """Visualize a segmentation mask with written labels."""
    if ax is None:
        ax = plt.gca()
    ax.imshow(mask, interpolation="nearest")
    ax.axis("off")

    for region in regionprops(mask):
        if region.label == 0:
            continue
        y, x = region.centroid
        ax.text(x, y, str(region.label), color="white", fontsize=10, ha="center", va="center")


def viz(img, flo):
    """Visualisation of flow, code from raft."""
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    flo = flo.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    plt.imshow(flo)
