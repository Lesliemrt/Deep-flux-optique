#!/usr/bin/env python
# noqa: EXE001
"""Example script to register two images with VoxelMorph models.

The source and target input images are expected to be affinely registered.
"""

# Core library imports
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

# Third-party imports
import torch

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa: PTH100, PTH118, PTH120
from utils.configs import DATA, DATA2, HEIGHT, WIDTH, device  # noqa: F401
from utils.utils import create_grid, interpole_and_pad, iou

import voxelmorph_folder.voxelmorph as vxm


def register(args):
    """Register two frames with a VoxelMorph model.

    Parameters
    ----------
    args : argparse.Namespace
        Commandline arguments.
    """
    print(f"Using device: {device}")  # noqa: T201

    # Load moving and fixed images
    moving = tiff.imread(args.moving)
    fixed = tiff.imread(args.fixed)

    # Normalize
    global_min = min(np.min(moving), np.min(fixed))
    global_max = max(np.max(moving), np.max(fixed))

    moving = (moving - global_min) / (global_max - global_min)
    fixed = (fixed - global_min) / (global_max - global_min)

    # Interpole/Pad images to HEIGHT, WIDTH if needed
    moving = interpole_and_pad(moving, [HEIGHT, WIDTH])
    fixed = interpole_and_pad(fixed, [HEIGHT, WIDTH])

    # Set up tensors
    input_moving = torch.from_numpy(moving).to(device).float().unsqueeze(0).unsqueeze(0)
    input_fixed = torch.from_numpy(fixed).to(device).float().unsqueeze(0).unsqueeze(0)

    # Create model
    model = vxm.nn.models.VxmPairwise(
        ndim=2,
        source_channels=1,
        target_channels=1,
        nb_features=[16, 16, 16, 16, 16],
        integration_steps=0,
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Predict
    with torch.no_grad():
        field, moved = model(
            source=input_moving, target=input_fixed, return_warped_source=True, return_field_type="displacement"
        )

    # Create grid for visualistion of the field
    grid = create_grid(moving.shape)
    tensor_grid = torch.from_numpy(grid).to(device).float().unsqueeze(0).unsqueeze(0)
    warped_grid = model.spatial_transformer(tensor_grid, field)

    results = {"fixed": fixed, "moving": moving, "warped_grid": warped_grid, "moved": moved}

    # If segmentations available, load them and warp the moving segmentation with the predicted field for evaluation
    if args.seg_available:
        # Load segmentation masks
        moving_path = args.moving.replace("02/t", "02_ERR_SEG/mask")
        fixed_path = args.fixed.replace("02/t", "02_ERR_SEG/mask")
        if os.path.isfile(moving_path) and os.path.isfile(fixed_path):  # noqa: PTH113
            seg_moving = tiff.imread(moving_path)
            seg_fixed = tiff.imread(fixed_path)
            seg_moving = interpole_and_pad(seg_moving, [HEIGHT, WIDTH])
            seg_fixed = interpole_and_pad(seg_fixed, [HEIGHT, WIDTH])
            # Warp moving segmentation with the predicted field
            tensor_seg_moving = torch.from_numpy(seg_moving).to(device).float().unsqueeze(0).unsqueeze(0)
            spatial_transformer = vxm.nn.modules.SpatialTransformer(interpolation_mode="nearest").to(device)
            warped_seg_moving = spatial_transformer(tensor_seg_moving, field)
            # Compute Dice score
            dice = vxm.py.utils.dice(warped_seg_moving.squeeze().cpu().numpy(), seg_fixed, include_zero=False)
            ref_dice = vxm.py.utils.dice(
                seg_moving, seg_fixed, include_zero=False
            )  # reference dice to compare with the dice after registration
            results["dice"] = np.mean(dice)
            results["ref_dice"] = np.mean(ref_dice)
            # Compute IoU score
            iou_score = iou(warped_seg_moving.squeeze().cpu().numpy(), seg_fixed, include_zero=False)
            ref_iou_score = iou(
                seg_moving, seg_fixed, include_zero=False
            )  # reference iou to compare with the iou after registration
            results["iou"] = np.mean(iou_score)
            results["ref_iou"] = np.mean(ref_iou_score)
        else:
            print("Segmentation files not found, skipping segmentation evaluation.")  # noqa: T201

    return results


if __name__ == "__main__":
    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving", default=f"{DATA}/02/t000.tif", help="moving image (source) filename")
    parser.add_argument("--fixed", default=f"{DATA}/02/t001.tif", help="fixed image (target) filename")
    parser.add_argument("--model", default="output/best.pt", help="weights of trained model to use for registration")
    parser.add_argument("--seg_available", default=True, help="whether segmentation is available for dice evaluation")
    # TODO : case 3D ?
    args = parser.parse_args()
    result = register(args)
    warped_grid = result["warped_grid"]
    moved = result["moved"]
    fixed = result["fixed"]
    moving = result["moving"]
    print(f"Dice score: {result['dice']}")  # noqa: T201
    print(f"Reference Dice score: {result['ref_dice']}")  # noqa: T201
    print(f"IoU score: {result['iou']}")  # noqa: T201
    print(f"Reference IoU score: {result['ref_iou']}")  # noqa: T201
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.title("Fixed")
    plt.imshow(fixed, cmap="gray")
    plt.subplot(2, 2, 2)
    plt.title("Moved")
    plt.imshow(moved.squeeze().cpu().numpy(), cmap="gray")
    plt.subplot(2, 2, 3)
    plt.title("Source")
    plt.imshow(moving, cmap="gray")
    plt.subplot(2, 2, 4)
    plt.title("Warped Grid")
    plt.imshow(warped_grid.squeeze().cpu().numpy(), cmap="gray")
    plt.show()
    # TODO save fig ?
