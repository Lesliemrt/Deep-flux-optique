# utils.py  # noqa: ERA001, INP001
import os
import sys

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
