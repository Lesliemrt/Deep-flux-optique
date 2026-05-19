# personnal tetst file # noqa: INP001
"""Example script to test registration of frames of a video with VoxelMorph models."""

# Core library imports
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

# Third-party imports
import torch

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa: PTH100, PTH118, PTH120
from train_cells import NormalizeMinMax, ThresholdTransform
from utils.configs import HEIGHT, TEST_DATA, TEST_SEG_DATA, WIDTH, device
from utils.utils import create_grid, interpole_and_pad, iou, normalisation_params, viz

import voxelmorph_folder.voxelmorph as vxm


def register_video(args):  # noqa: PLR0915
    """Register video frames with a VoxelMorph model.

    Parameters
    ----------
    args : argparse.Namespace
        Commandline arguments.
    """
    print(f"Using device: {device}")  # noqa: T201

    # Load video and segmentations frames
    frames_path_list = sorted(Path(args.video_path).iterdir())
    seg_path_list = sorted(Path(args.seg_path).iterdir()) if args.seg_path is not None else None
    global_min, global_max = normalisation_params(frames_path_list)

    # Create model
    model = vxm.nn.models.VxmPairwise(
        ndim=2,
        source_channels=1,
        target_channels=1,
        nb_features=[16, 16, 16, 16, 16],
        integration_steps=0,
    ).to(device)
    model_path = f"{args.output_dir}/best.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Load model : {model_path}")  # noqa: T201
    model.eval()

    # Create grid for visualisation of the field
    grid = create_grid([HEIGHT, WIDTH], spacing=20)
    tensor_grid = torch.from_numpy(grid).to(device).float().unsqueeze(0).unsqueeze(0)

    results = []
    for idx in range(len(frames_path_list) - args.gap):
        # Load moving and fixed images
        idx2 = idx + args.gap
        moving = tiff.imread(frames_path_list[idx])
        fixed = tiff.imread(frames_path_list[idx2])

        # Interpole/Pad images to HEIGHT, WIDTH if needed
        moving = interpole_and_pad(moving, [HEIGHT, WIDTH])
        fixed = interpole_and_pad(fixed, [HEIGHT, WIDTH])

        # To tensor
        input_moving = torch.from_numpy(moving).float().to(device).unsqueeze(0).unsqueeze(0)
        input_fixed = torch.from_numpy(fixed).float().to(device).unsqueeze(0).unsqueeze(0)

        # Normalisation
        normalise = NormalizeMinMax(global_min=global_min, global_max=global_max)
        input_moving = normalise(input_moving)
        input_fixed = normalise(input_fixed)

        # Binarise
        with Path(f"{args.output_dir}/config.json").open() as j:
            config = json.load(j)
            if config["binarization"]:
                thr = 0.05
                binarise = ThresholdTransform(thr_1=thr)
                input_moving = binarise(input_moving)
                input_fixed = binarise(input_fixed)

        # Predict
        with torch.no_grad():
            field, moved = model(
                source=input_moving, target=input_fixed, return_warped_source=True, return_field_type="displacement"
            )

        # Apply warp on grid for visualistion of the field
        warped_grid = model.spatial_transformer(tensor_grid, field)

        results_pair = {
            "fixed": input_fixed,
            "moving": input_moving,
            "warped_grid": warped_grid,
            "moved": moved,
            "field": field,
        }

        # If segmentations available, load them and warp the moving segmentation with the predicted field for evaluation
        if args.seg_path is not None:
            # Load segmentation masks
            seg_moving = tiff.imread(seg_path_list[idx])
            seg_fixed = tiff.imread(seg_path_list[idx2])
            seg_moving = interpole_and_pad(seg_moving, [HEIGHT, WIDTH])
            seg_fixed = interpole_and_pad(seg_fixed, [HEIGHT, WIDTH])
            # Take only pairs of frame with same nbr of cells (to have max dice = 1 in case of disappaering cell in vid)
            moving_labels = set(np.unique([seg_moving]))
            fixed_labels = set(np.unique([seg_fixed]))
            if moving_labels != fixed_labels:
                # print(f"ignored frame {idx} and {idx2} : {moving_labels} vs {fixed_labels}")  # noqa: ERA001
                continue
            # Warp moving segmentation with the predicted field
            tensor_seg_moving = torch.from_numpy(seg_moving).to(device).float().unsqueeze(0).unsqueeze(0)
            spatial_transformer = vxm.nn.modules.SpatialTransformer(interpolation_mode="nearest").to(device)
            warped_seg_moving = spatial_transformer(tensor_seg_moving, field)
            # Compute Dice score
            dice = vxm.py.utils.dice(warped_seg_moving.squeeze().cpu().numpy(), seg_fixed, include_zero=False)
            ref_dice = vxm.py.utils.dice(
                seg_moving, seg_fixed, include_zero=False
            )  # reference dice to compare with the dice after registration
            results_pair["dice"] = np.mean(dice)
            results_pair["ref_dice"] = np.mean(ref_dice)
            # Compute IoU score
            iou_score = iou(warped_seg_moving.squeeze().cpu().numpy(), seg_fixed, include_zero=False)
            ref_iou_score = iou(
                seg_moving, seg_fixed, include_zero=False
            )  # reference iou to compare with the iou after registration
            results_pair["iou"] = np.mean(iou_score)
            results_pair["ref_iou"] = np.mean(ref_iou_score)
        else:
            print("Segmentation files not found, skipping segmentation evaluation.")  # noqa: T201
        results.append(results_pair)

    return results


if __name__ == "__main__":
    # Parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default=f"{TEST_DATA}", help="Moving image (source) filename")
    parser.add_argument("--gap", default=10, help="Temporal gap between 2 frames")
    parser.add_argument(
        "--seg_path",
        default=f"{TEST_SEG_DATA}",
        help="Segmentations for evaluation (if segmentations not available, set to None)",
    )
    parser.add_argument(
        "--output_dir", default="outputs/output_20260519_145503", help="Directory to find model and to save results"
    )
    args = parser.parse_args()

    # Compute results for all pairs of frames in the video
    results = register_video(args)
    # Create results directory if it doesn't exist
    results_dir = Path(f"{args.output_dir}/results_gap{args.gap}")
    results_dir.mkdir(parents=True, exist_ok=True)
    # Compute de global dice score across all pairs of frames if segmentations are available
    if "dice" in results[0]:
        global_dice = np.mean([result["dice"] for result in results])
        global_ref_dice = np.mean([result["ref_dice"] for result in results])
        print(f"Global Dice score across all pairs of frames: {global_dice}")  # noqa: T201
        print(f"Global reference Dice score across all pairs of frames: {global_ref_dice}")  # noqa: T201
        print(f"Global Dice : {[f'{result[('dice')]:.4f}' for result in results]}")  # noqa: T201
        # Compute de global iou score
        global_iou = np.mean([result["iou"] for result in results])
        global_ref_iou = np.mean([result["ref_iou"] for result in results])
        print(f"Global IoU score across all pairs of frames: {global_iou}")  # noqa: T201
        print(f"Global reference IoU score across all pairs of frames: {global_ref_iou}")  # noqa: T201
        print(f"Global IoU : {[f'{result[('iou')]:.4f}' for result in results]}")  # noqa: T201
        # Save metrics in a json file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
        metrics = {
            "data": args.video_path,
            "date": timestamp,
            "model": args.output_dir,
            "device": str(device),
            "gap": args.gap,
            "global_dice": global_dice,
            "global_ref_dice": global_ref_dice,
            "global_iou": global_iou,
            "global_ref_iou": global_ref_iou,
            "dice_per_pair": [result["dice"] for result in results],
            "ref_dice_per_pair": [result["ref_dice"] for result in results],
            "iou_per_pair": [result["iou"] for result in results],
            "ref_iou_per_pair": [result["ref_iou"] for result in results],
        }
        with Path.open(results_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
    # Visualise results for first 2 frames
    idx = 0
    warped_grid = results[idx]["warped_grid"]
    field = results[idx]["field"]
    moved = results[idx]["moved"]
    fixed = results[idx]["fixed"]
    moving = results[idx]["moving"]
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.title("Fixed")
    plt.imshow(fixed.squeeze().cpu().numpy(), cmap="gray")
    plt.subplot(2, 2, 2)
    plt.title("Moved")
    plt.imshow(moved.squeeze().cpu().numpy(), cmap="gray")
    plt.subplot(2, 2, 3)
    plt.title("Moving")
    plt.imshow(moving.squeeze().cpu().numpy(), cmap="gray")
    plt.subplot(2, 2, 4)
    plt.title("Flow")
    viz(img=fixed, flo=field)
    plt.savefig(results_dir / f"registration_result_for_paire_{idx}.png")
