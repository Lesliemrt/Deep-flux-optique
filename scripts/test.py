# personnal tetst file # noqa: INP001
"""Script to test registration of frames of a video with VoxelMorph models.
Method with testloader and register_cells (but slower than tests.py)."""

# Core library imports
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from argparse import Namespace

# Third-party imports
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raft", "core"))
)  # noqa: PTH100, PTH118, PTH120
from my_utils.configs import (
    TEST_DATA,  # noqa: F401
    TEST_DATA_HELA,  # noqa: F401
    TEST_DATA_HSC,  # noqa: F401
    TEST_DATA_MUSC,  # noqa: F401
    TEST_SEG_DATA,  # noqa: F401
    TEST_SEG_DATA_HELA,  # noqa: F401
    TEST_SEG_DATA_HSC,  # noqa: F401
    TEST_SEG_DATA_MUSC,  # noqa: F401
    device,
)
from dataset import VxmVideoDataset
from my_utils.utils import iou, viz
from flow_visualizer import FlowVisualizer
from raft import RAFT
from register_cells import (
    register_pair,
    register_pair_tvl1,
    register_pair_raft,
    register_pair_fb,
)

import voxelmorph_folder.voxelmorph as vxm


def evaluate_registration(segmentation_moving, segmentation_fixed, field):
    """Evaluate the registration results."""
    score = {}
    # Load segmentation masks
    seg_moving = segmentation_moving
    seg_fixed = segmentation_fixed
    # Take only pairs of frame with same nbr of cells (to have max dice = 1 in case of disappaering cell in vid)
    moving_labels = set(np.unique([seg_moving]))
    fixed_labels = set(np.unique([seg_fixed]))
    if moving_labels != fixed_labels:
        return None
    # Warp moving segmentation with the predicted field
    tensor_seg_moving = torch.from_numpy(seg_moving).to(device).float().unsqueeze(0)
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

    # To visulaise seg evaluation
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    plt.subplot(2, 2, 1)
    ax[0, 0].set_title("Fixed segmentation")
    ax[0, 0].imshow(np.squeeze(seg_fixed), cmap="gray")
    plt.subplot(2, 2, 2)
    ax[0, 1].set_title("Warped moving segmentation")
    ax[0, 1].imshow(warped_seg_moving.squeeze().cpu().numpy(), cmap="gray")
    plt.subplot(2, 2, 3)
    ax[1, 0].set_title("Moving segmentation")
    ax[1, 0].imshow(np.squeeze(seg_moving), cmap="gray")
    plt.subplot(2, 2, 4)
    ax[1, 1].set_title("Flow")
    viz(flo=field, ax=ax[1, 1])
    return score, fig


def create_model(args):
    """Create the model for registration."""
    if args.model_type == "voxelmorph":
        model = vxm.nn.models.VxmPairwise(
            ndim=2,
            source_channels=1,
            target_channels=1,
            nb_features=[16, 16, 16, 16, 16],
            integration_steps=0,
            device=device,
        ).to(device)
        model_path = f"{args.output_dir}/best.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Load model : {model_path}")  # noqa: T201
        model.eval()
        if mlflow.active_run():
            mlflow.log_param("model_path", model_path)
    elif args.model_type == "tvl1":
        model = None
    elif args.model_type == "raft":
        raft_args = Namespace(
            model="raft-things.pth",
            path=args.video_path,
            small=False,
            mixed_precision=False,
            alternate_corr=False,
        )
        model = torch.nn.DataParallel(RAFT(raft_args))
        model_path = "raft_models/raft-things.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))

        model = model.module
        model.to(device)
        model.eval()
    elif args.model_type == "farnback":
        model = cv2.FarnebackOpticalFlow.create()
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    return model


def register_video(args):
    """Register video frames.

    Parameters
    ----------
    args : argparse.Namespace
        Commandline arguments.
    """
    print(f"Using device: {device}")  # noqa: T201

    # MLFLOW
    if args.model_type == "voxelmorph":
        with Path(f"{args.output_dir}/config.json").open() as j:
            config = json.load(j)  # To get informations from vxm training
    else:
        config = {}
        args.output_dir = f"outputs/{args.model_type}"

    if args.mlflow:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        parent_run_id = config.get("mlflow_run_id", None)
        mlflow.start_run(
            run_name=f"Test gap {args.gap} model type {args.model_type} on {args.video_path}",
            nested=True,
            parent_run_id=parent_run_id,  # To link the test run to the training run (if not specified, the test run will be a independant run)
        )
        print("MLflow tracking enabled.")  # noqa: T201
        mlflow.log_params(vars(args))

    # 1. Create dataloader
    test_dataset = VxmVideoDataset(
        video_path=args.video_path,
        seg_path=args.seg_path,
        seg_type=None,
        mode="test",
        gap=args.gap,
        multiplier=1,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,  # To speed up, if not enough RAM, set to False
    )

    # 2. Create model
    model = create_model(args)

    # Create results directory if it doesn't exist
    results_dir = Path(f"{args.output_dir}/results_gap{args.gap}")
    results_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = results_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    results = []
    flow_maps = {}
    video = []  # list of frames to visualize the flow maps on (only for gap=1 and if args.visualize is True)
    for batch in tqdm(test_loader, desc="Registering pairs of frames"):
        # Load moving and fixed images
        moving = batch["source"].to(device).unsqueeze(0)
        fixed = batch["target"].to(device).unsqueeze(0)
        idx = batch["idx1"].numpy().item()
        idx2 = batch["idx2"].numpy().item()

        # 3. Compute registration
        if args.model_type == "voxelmorph":
            field, fig, input_moving, input_fixed = register_pair(moving, fixed, model)
        elif args.model_type == "tvl1":
            field, fig, input_moving, input_fixed = register_pair_tvl1(moving, fixed)
        elif args.model_type == "raft":
            field, fig, input_moving, input_fixed = register_pair_raft(
                moving, fixed, model
            )
        elif args.model_type == "farnback":
            field, fig, input_moving, input_fixed = register_pair_fb(
                moving, fixed, model
            )
        fig.savefig(viz_dir / f"registration_result_for_paire_{idx}-{idx2}.png")
        plt.close(fig)

        if args.visualize:
            flow_maps[(idx2, idx)] = field.squeeze().cpu().numpy()
            if idx == 0:
                video.append(
                    np.expand_dims(input_moving, axis=-1)
                )  # channel dimension needed for visualization
            video.append(
                np.expand_dims(input_fixed, axis=-1)
            )  # channel dimension neededfor visualization
            if args.model_type == "voxelmorph":
                inv_field, _, _, _ = register_pair(fixed, moving, model)
            elif args.model_type == "tvl1":
                inv_field, _, _, _ = register_pair_tvl1(fixed, moving)
            elif args.model_type == "raft":
                inv_field, _, _, _ = register_pair_raft(fixed, moving, model)
            elif args.model_type == "farnback":
                inv_field, _, _, _ = register_pair_fb(fixed, moving, model)
            flow_maps[(idx, idx2)] = inv_field.squeeze().cpu().numpy()

        if mlflow.active_run():
            mlflow.log_figure(
                fig, artifact_file=f"viz/registration_result_for_paire_{idx}-{idx2}.png"
            )

        # 4. Evaluation for the pair
        # If segmentations available, load them and warp the moving segmentation with the predicted field for evaluation
        if args.seg_path is not None:
            # Load segmentation masks
            seg_moving = batch["source_seg"].numpy()
            seg_fixed = batch["target_seg"].numpy()
            res = evaluate_registration(seg_moving, seg_fixed, field)
            if res is None:
                continue
            score_pair, fig = res
            results.append(score_pair)
            fig.savefig(viz_dir / f"seg_eval_for_paire_{idx}-{idx2}.png")
            if mlflow.active_run():
                mlflow.log_figure(
                    fig, artifact_file=f"seg_eval_for_paire_{idx}-{idx2}.png"
                )
            plt.close(fig)

        else:
            print("Segmentation files not found, skipping segmentation evaluation.")  # noqa: T201

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
        # TODO fonction pour faire ça pck la c'est moche
        dice_per_pair = [result["dice"] for result in results if result is not None]
        ref_dice_scores_per_pair = [
            result["ref_dice"] for result in results if result is not None
        ]
        iou_scores_per_pair = [
            result["iou"] for result in results if result is not None
        ]
        ref_iou_scores_per_pair = [
            result["ref_iou"] for result in results if result is not None
        ]
        global_dice = np.mean(dice_per_pair)
        global_ref_dice = np.mean(ref_dice_scores_per_pair)
        global_iou = np.mean(iou_scores_per_pair)
        global_ref_iou = np.mean(ref_iou_scores_per_pair)
        # Save metrics in a json file
        metrics["global_dice"] = global_dice
        metrics["global_ref_dice"] = global_ref_dice
        metrics["global_iou"] = global_iou
        metrics["global_ref_iou"] = global_ref_iou
        metrics["dice_per_pair"] = dice_per_pair
        metrics["ref_dice_per_pair"] = ref_dice_scores_per_pair
        metrics["iou_per_pair"] = iou_scores_per_pair
        metrics["ref_iou_per_pair"] = ref_iou_scores_per_pair
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

    if args.visualize:
        visualizer = FlowVisualizer(video, flow_maps, grid_step=20, gap=args.gap)
        visualizer.run()


if __name__ == "__main__":
    # Parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlflow", action="store_true", help="Enable mlflow tracking during training"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization of flow maps",
    )
    parser.add_argument(
        "--video_path",
        default=f"{TEST_DATA_MUSC}",
        help="Moving image (source) filename",
    )
    parser.add_argument("--gap", default=1, help="Temporal gap between 2 frames")
    parser.add_argument(
        "--seg_path",
        default=TEST_SEG_DATA_MUSC,
        help="Segmentations for evaluation (if segmentations not available, set to None)",
    )
    parser.add_argument(
        "--model_type",
        default="voxelmorph",
        choices=["voxelmorph", "tvl1", "raft", "farnback"],
        help="Type of model to use for registration",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/output_20260611_105251",
        help="Directory to find model and to save results",
    )
    args = parser.parse_args()

    register_video(args)
