#!/usr/bin/env python noqa: EXE001
"""Example script to register two images with VoxelMorph models.

The source and target input images are expected to be affinely registered.
"""  # noqa: EXE001

# Core library imports
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from skimage.registration import optical_flow_tvl1
from PIL import Image

# Third-party imports
import torch

# Local imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raft", "core"))
)  # noqa: PTH100, PTH118, PTH120
from train_cells import NormalizeMinMax, ThresholdTransform
from my_utils.configs import HEIGHT, WIDTH, device
from my_utils.utils import interpole_and_pad, viz, normalise

import voxelmorph_folder.voxelmorph as vxm


def register_pair(moving, fixed, model_type, model, global_min, global_max, config):
    """Register a pair of frames using Voxelmorph model.

    Parameters
    ----------
    moving : numpy.ndarray
        Moving image.
    fixed : numpy.ndarray
        Fixed image.
    model_type : str
        Type of model to use for registration.
    model : torch.nn.Module
        Model to use for registration.
    global_min : float
        Global minimum for normalization.
    global_max : float
        Global maximum for normalization.
    config : dict
        Configuration dictionary.

    Returns
    -------
    field : torch.Tensor
        Predicted deformation field.
    fig : matplotlib.figure.Figure
        Figure showing the fixed image, the moved image, the moving image and the flow field.
    input_moving : numpy.ndarray
        Normalized moving image.
    input_fixed : numpy.ndarray
        Normalized fixed image.
    """
    # Interpole/Pad images to HEIGHT, WIDTH if needed
    moving_resized = interpole_and_pad(moving, [HEIGHT, WIDTH])
    fixed_resized = interpole_and_pad(fixed, [HEIGHT, WIDTH])

    # To tensor
    input_moving = (
        torch.from_numpy(moving_resized).float().to(device).unsqueeze(0).unsqueeze(0)
    )
    input_fixed = (
        torch.from_numpy(fixed_resized).float().to(device).unsqueeze(0).unsqueeze(0)
    )

    # Normalisation
    normalise = NormalizeMinMax(global_min=global_min, global_max=global_max)
    input_moving = normalise(input_moving)
    input_fixed = normalise(input_fixed)

    # Binarise
    if config["binarization"]:
        thr = 0.05
        binarise = ThresholdTransform(thr_1=thr)
        input_moving = binarise(input_moving)
        input_fixed = binarise(input_fixed)

    # Predict
    with torch.no_grad():
        field, moved = model(
            source=input_moving,
            target=input_fixed,
            return_warped_source=True,
            return_field_type="displacement",
        )

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    plt.subplot(2, 2, 1)
    ax[0, 0].set_title("Fixed")
    ax[0, 0].imshow(input_fixed.squeeze().cpu().numpy(), cmap="gray")
    plt.subplot(2, 2, 2)
    ax[0, 1].set_title("Moved")
    ax[0, 1].imshow(moved.squeeze().cpu().numpy(), cmap="gray")
    plt.subplot(2, 2, 3)
    ax[1, 0].set_title("Moving")
    ax[1, 0].imshow(input_moving.squeeze().cpu().numpy(), cmap="gray")
    plt.subplot(2, 2, 4)
    ax[1, 1].set_title("Flow")
    viz(flo=field, ax=ax[1, 1])

    return (
        field,
        fig,
        input_moving.squeeze().cpu().numpy(),
        input_fixed.squeeze().cpu().numpy(),
    )


def register_pair_tvl1(moving, fixed, global_min, global_max):
    """Register a pair of frames using TVL1 optical flow model.

    Parameters
    ----------
    moving : numpy.ndarray
        Moving image.
    fixed : numpy.ndarray
        Fixed image.
    global_min : float
        Global minimum for normalization.
    global_max : float
        Global maximum for normalization.

    Returns
    -------
    field : torch.Tensor
        Predicted deformation field.
    fig : matplotlib.figure.Figure
        Figure showing the fixed image, the moved image, the moving image and the flow field.
    input_moving : numpy.ndarray
        Normalized moving image.
    input_fixed : numpy.ndarray
        Normalized fixed image.
    """
    # Interpole/Pad images to HEIGHT, WIDTH if needed
    moving_resized = interpole_and_pad(moving, [HEIGHT, WIDTH])
    fixed_resized = interpole_and_pad(fixed, [HEIGHT, WIDTH])

    # Normalisation
    input_moving = normalise(moving_resized, global_min, global_max)
    input_fixed = normalise(fixed_resized, global_min, global_max)

    # To tensor
    tensor_moving = (
        torch.from_numpy(input_moving).float().to(device).unsqueeze(0).unsqueeze(0)
    )

    # Predict
    field = optical_flow_tvl1(
        reference_image=input_fixed, moving_image=input_moving
    )  # TODO : verifier s'il faut inverser images pour avoir field dans le bon sens
    spatial_transformer = vxm.nn.modules.SpatialTransformer(
        interpolation_mode="bilinear"
    ).to(device)
    tensor_field = torch.from_numpy(field).float().to(device).unsqueeze(0)
    moved = spatial_transformer(tensor_moving, tensor_field)
    #  moving_image : torch.Tensor shape: (B, C, *spatial_dims), deformation_field : torch.Tensor shape: (B, ndim, *spatial_dims).

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    plt.subplot(2, 2, 1)
    ax[0, 0].set_title("Fixed")
    ax[0, 0].imshow(input_fixed, cmap="gray")
    plt.subplot(2, 2, 2)
    ax[0, 1].set_title("Moved")
    ax[0, 1].imshow(moved.squeeze().cpu().numpy(), cmap="gray")
    plt.subplot(2, 2, 3)
    ax[1, 0].set_title("Moving")
    ax[1, 0].imshow(input_moving, cmap="gray")
    plt.subplot(2, 2, 4)
    ax[1, 1].set_title("Flow")
    viz(flo=tensor_field, ax=ax[1, 1])

    return tensor_field, fig, input_moving, input_fixed


def register_pair_raft(moving_path, fixed_path, model, global_min, global_max):
    """Register a pair of frames using RAFT optical flow model.

    Parameters
    ----------
    moving_path : str
        Path to the moving image.
    fixed_path : str
        Path to the fixed image.
    model : torch.nn.Module
        RAFT optical flow model.
    global_min : float
        Global minimum for normalization.
    global_max : float
        Global maximum for normalization.

    Returns
    -------
    field : torch.Tensor
        Predicted deformation field.
    fig : matplotlib.figure.Figure
        Figure showing the fixed image, the moved image, the moving image and the flow field.
    input_moving : numpy.ndarray
        Normalized moving image.
    input_fixed : numpy.ndarray
        Normalized fixed image.
    """

    moving = np.array(Image.open(moving_path).convert("RGB")).astype(np.uint8)
    fixed = np.array(Image.open(fixed_path).convert("RGB")).astype(np.uint8)

    # Interpole/Pad images to HEIGHT, WIDTH if needed
    # moving_resized = interpole_and_pad(moving, [HEIGHT, WIDTH])
    # fixed_resized = interpole_and_pad(fixed, [HEIGHT, WIDTH])

    # To tensor
    input_moving = (
        torch.from_numpy(moving).permute(2, 0, 1).float().unsqueeze(0).to(device)
    )
    input_fixed = (
        torch.from_numpy(fixed).permute(2, 0, 1).float().unsqueeze(0).to(device)
    )

    # Normalisation
    normalise = NormalizeMinMax(global_min=global_min, global_max=global_max)
    input_moving = normalise(input_moving)
    input_fixed = normalise(input_fixed)

    # Predict
    with torch.no_grad():
        _, tensor_field = model(
            input_fixed,  # to get flow in the direction fixed -> moving (like voxelmorph)
            input_moving,
            iters=20,
            test_mode=True,
        )
    tensor_field = torch.flip(tensor_field, dims=[1])

    spatial_transformer = vxm.nn.modules.SpatialTransformer(
        interpolation_mode="bilinear"
    ).to(device)
    moved = spatial_transformer(input_moving, tensor_field)

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    plt.subplot(2, 2, 1)
    ax[0, 0].set_title("Fixed")
    ax[0, 0].imshow(input_fixed.squeeze().cpu().numpy().mean(axis=0), cmap="gray")
    plt.subplot(2, 2, 2)
    ax[0, 1].set_title("Moved")
    ax[0, 1].imshow(moved.squeeze().cpu().numpy().mean(axis=0), cmap="gray")
    plt.subplot(2, 2, 3)
    ax[1, 0].set_title("Moving")
    ax[1, 0].imshow(input_moving.squeeze().cpu().numpy().mean(axis=0), cmap="gray")
    plt.subplot(2, 2, 4)
    ax[1, 1].set_title("Flow")
    viz(flo=tensor_field, ax=ax[1, 1])

    return (
        tensor_field,
        fig,
        input_moving.squeeze().cpu().numpy().mean(axis=0),
        input_fixed.squeeze().cpu().numpy().mean(axis=0),
    )


def register_pair_fb(moving, fixed, model, global_min, global_max):
    """Register a pair of frames using TVL1 optical flow model.

    Parameters
    ----------
    moving : numpy.ndarray
        Moving image.
    fixed : numpy.ndarray
        Fixed image.
    model : cv2.FarnebackOpticalFlow_create
        Optical flow model.
    global_min : float
        Global minimum for normalization.
    global_max : float
        Global maximum for normalization.
    config : dict
        Configuration dictionary.

    Returns
    -------
    field : torch.Tensor
        Predicted deformation field.
    fig : matplotlib.figure.Figure
        Figure showing the fixed image, the moved image, the moving image and the flow field.
    input_moving : numpy.ndarray
        Normalized moving image.
    input_fixed : numpy.ndarray
        Normalized fixed image.
    """
    # Interpole/Pad images to HEIGHT, WIDTH if needed
    # moving_resized = interpole_and_pad(moving, [HEIGHT, WIDTH])
    # fixed_resized = interpole_and_pad(fixed, [HEIGHT, WIDTH])

    # Normalisation
    input_moving = moving.astype(np.uint8)
    input_fixed = fixed.astype(np.uint8)

    # To tensor
    tensor_moving = (
        torch.from_numpy(input_moving).float().to(device).unsqueeze(0).unsqueeze(0)
    )

    # Predict
    field = model.calc(input_fixed, input_moving, None)
    spatial_transformer = vxm.nn.modules.SpatialTransformer(
        interpolation_mode="bilinear"
    ).to(device)
    tensor_field = (
        torch.from_numpy(field).float().to(device).unsqueeze(0).permute(0, 3, 1, 2)
    )  # from (B, H, W, ndim) to (B, ndim, H, W)
    moved = spatial_transformer(tensor_moving, tensor_field)
    #  moving_image : torch.Tensor shape: (B, C, *spatial_dims), deformation_field : torch.Tensor shape: (B, ndim, *spatial_dims).

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    plt.subplot(2, 2, 1)
    ax[0, 0].set_title("Fixed")
    ax[0, 0].imshow(input_fixed, cmap="gray")
    plt.subplot(2, 2, 2)
    ax[0, 1].set_title("Moved")
    ax[0, 1].imshow(moved.squeeze().cpu().numpy(), cmap="gray")
    plt.subplot(2, 2, 3)
    ax[1, 0].set_title("Moving")
    ax[1, 0].imshow(input_moving, cmap="gray")
    plt.subplot(2, 2, 4)
    ax[1, 1].set_title("Flow")
    viz(
        flo=tensor_field,
        ax=ax[1, 1],
    )

    return tensor_field, fig, input_moving, input_fixed
