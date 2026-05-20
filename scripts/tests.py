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
import mlflow
import numpy as np
import tifffile as tiff
from skimage.registration import optical_flow_tvl1

# Third-party imports
import torch

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa: PTH100, PTH118, PTH120
from train_cells import NormalizeMinMax, ThresholdTransform
from utils.configs import HEIGHT, TEST_DATA, TEST_SEG_DATA, WIDTH, device
from utils.utils import interpole_and_pad, iou, normalisation_params, viz, normalise

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
    viz(img=input_fixed, flo=field, ax=ax[1, 1])

    return field, fig


def register_pair_tvl1(moving, fixed, model_type, global_min, global_max, config):
    """Register a pair of frames using TVL1 optical flow model.

    Parameters
    ----------
    moving : numpy.ndarray
        Moving image.
    fixed : numpy.ndarray
        Fixed image.
    model_type : str
        Type of model to use for registration.
    global_min : float
        Global minimum for normalization.
    global_max : float
        Global maximum for normalization.
    config : dict
        Configuration dictionary.
    """
    # Interpole/Pad images to HEIGHT, WIDTH if needed
    moving_resized = interpole_and_pad(moving, [HEIGHT, WIDTH])
    fixed_resized = interpole_and_pad(fixed, [HEIGHT, WIDTH])

    # Normalisation
    input_moving = normalise(moving_resized, global_min, global_max)
    input_fixed = normalise(fixed_resized, global_min, global_max)

    # Predict
    field = optical_flow_tvl1(reference_image=input_fixed, moving_image=input_moving)
    spatial_transformer = vxm.nn.modules.SpatialTransformer(
        interpolation_mode="bilinear"
    ).to(device)
    tensor_moving = (
        torch.from_numpy(input_moving).float().to(device).unsqueeze(0).unsqueeze(0)
    )
    tensor_field = torch.from_numpy(field).float().to(device).unsqueeze(0)
    moved = spatial_transformer(tensor_moving, tensor_field)
    #  moving_image : torch.Tensor
    #         Tensor to be spatially transformed by `deformation_field`.
    #         Shape: (B, C, *spatial_dims).
    #     deformation_field : torch.Tensor
    #         Field causing the spatial transformation of `moving_image`.
    #         Shape: (B, ndim, *spatial_dims).

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
        img=torch.from_numpy(input_fixed).float().to(device).unsqueeze(0).unsqueeze(0),
        flo=tensor_field,
        ax=ax[1, 1],
    )

    return tensor_field, fig


def evaluate_registration(segmentation_moving, segmentation_fixed, field):
    """Evaluate the registration results."""
    score = {}
    # Load segmentation masks
    seg_moving = interpole_and_pad(segmentation_moving, [HEIGHT, WIDTH])
    seg_fixed = interpole_and_pad(segmentation_fixed, [HEIGHT, WIDTH])
    # Take only pairs of frame with same nbr of cells (to have max dice = 1 in case of disappaering cell in vid)
    moving_labels = set(np.unique([seg_moving]))
    fixed_labels = set(np.unique([seg_fixed]))
    if moving_labels != fixed_labels:
        return None
    # Warp moving segmentation with the predicted field
    tensor_seg_moving = (
        torch.from_numpy(seg_moving).to(device).float().unsqueeze(0).unsqueeze(0)
    )
    spatial_transformer = vxm.nn.modules.SpatialTransformer(
        interpolation_mode="nearest"
    ).to(device)
    warped_seg_moving = spatial_transformer(tensor_seg_moving, field)
    # Compute Dice score
    dice = vxm.py.utils.dice(
        warped_seg_moving.squeeze().cpu().numpy(), seg_fixed, include_zero=False
    )
    ref_dice = vxm.py.utils.dice(
        seg_moving, seg_fixed, include_zero=False
    )  # reference dice to compare with the dice after registration
    score["dice"] = np.mean(dice)
    score["ref_dice"] = np.mean(ref_dice)
    # Compute IoU score
    iou_score = iou(
        warped_seg_moving.squeeze().cpu().numpy(), seg_fixed, include_zero=False
    )
    ref_iou_score = iou(
        seg_moving, seg_fixed, include_zero=False
    )  # reference iou to compare with the iou after registration
    score["iou"] = np.mean(iou_score)
    score["ref_iou"] = np.mean(ref_iou_score)
    return score


def register_video(args):
    """Register video frames.

    Parameters
    ----------
    args : argparse.Namespace
        Commandline arguments.
    """
    print(f"Using device: {device}")  # noqa: T201

    # 1. Load video and segmentations frames
    frames_path_list = sorted(Path(args.video_path).iterdir())
    seg_path_list = (
        sorted(Path(args.seg_path).iterdir()) if args.seg_path is not None else None
    )
    global_min, global_max = normalisation_params(frames_path_list)

    if args.model_type == "voxelmorph":
        # Open config file
        with Path(f"{args.output_dir}/config.json").open() as j:
            config = json.load(j)
    else:
        config = {}

    # MLFLOW
    if args.mlflow:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        parent_run_id = config.get("mlflow_run_id", None)
        mlflow.start_run(
            run_name=f"Test gap {args.gap} model type {args.model_type} on {args.video_path}",
            nested=True,
            parent_run_id=parent_run_id,
        )
        print("MLflow tracking enabled.")  # noqa: T201
        mlflow.log_params(vars(args))

    # 2. Create model
    if args.model_type == "voxelmorph":
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
        if mlflow.active_run():
            mlflow.log_param("model_path", model_path)

    # Create results directory if it doesn't exist
    results_dir = Path(f"{args.output_dir}/results_gap{args.gap}")
    results_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = results_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for idx in range(len(frames_path_list) - args.gap):
        # Load moving and fixed images
        idx2 = idx + args.gap
        moving = tiff.imread(frames_path_list[idx])
        fixed = tiff.imread(frames_path_list[idx2])

        # 3. Compute registration
        if args.model_type == "voxelmorph":
            field, fig = register_pair(
                moving, fixed, args.model_type, model, global_min, global_max, config
            )
        elif args.model_type == "tvl1":
            field, fig = register_pair_tvl1(
                moving, fixed, args.model_type, global_min, global_max, config
            )
        fig.savefig(viz_dir / f"registration_result_for_paire_{idx}.png")
        plt.close(fig)

        if mlflow.active_run():
            mlflow.log_figure(
                fig, artifact_file=f"viz/registration_result_for_paire_{idx}.png"
            )

        # 4. Evaluation for the pair
        # If segmentations available, load them and warp the moving segmentation with the predicted field for evaluation
        if args.seg_path is not None:
            # Load segmentation masks
            seg_moving = tiff.imread(seg_path_list[idx])
            seg_fixed = tiff.imread(seg_path_list[idx2])
            score_pair = evaluate_registration(seg_moving, seg_fixed, field)
            if score_pair is None:
                continue
        else:
            print("Segmentation files not found, skipping segmentation evaluation.")  # noqa: T201

        results.append(score_pair)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    metrics = {
        "data": args.video_path,
        "date": timestamp,
        "model": args.output_dir,
        "device": str(device),
        "gap": args.gap,
    }
    # 5. Global evaluation
    # Compute de global dice score an iou score
    if args.seg_path is not None:
        global_dice = np.mean([result["dice"] for result in results])
        global_ref_dice = np.mean([result["ref_dice"] for result in results])
        # Compute de global iou score
        global_iou = np.mean([result["iou"] for result in results])
        global_ref_iou = np.mean([result["ref_iou"] for result in results])
        # Save metrics in a json file
        metrics["global_dice"] = global_dice
        metrics["global_ref_dice"] = global_ref_dice
        metrics["global_iou"] = global_iou
        metrics["global_ref_iou"] = global_ref_iou
        metrics["dice_per_pair"] = [result["dice"] for result in results]
        metrics["ref_dice_per_pair"] = [result["ref_dice"] for result in results]
        metrics["iou_per_pair"] = [result["iou"] for result in results]
        metrics["ref_iou_per_pair"] = [result["ref_iou"] for result in results]
        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "global_dice": global_dice,
                    "global_ref_dice": global_ref_dice,
                    "global_iou": global_iou,
                    "global_ref_iou": global_ref_iou,
                }
            )
    with Path.open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    if args.mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    # Parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlflow", action="store_true", help="Enable mlflow tracking during training"
    )
    parser.add_argument(
        "--video_path", default=f"{TEST_DATA}", help="Moving image (source) filename"
    )
    parser.add_argument("--gap", default=10, help="Temporal gap between 2 frames")
    parser.add_argument(
        "--seg_path",
        default=f"{TEST_SEG_DATA}",
        help="Segmentations for evaluation (if segmentations not available, set to None)",
    )
    parser.add_argument(
        "--model_type",
        default="voxelmorph",
        choices=["voxelmorph", "tvl1"],
        help="Type of model to use for registration",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/TVL1",
        help="Directory to find model and to save results",
    )
    args = parser.parse_args()

    register_video(args)
