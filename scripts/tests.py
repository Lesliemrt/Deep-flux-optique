# personnal tetst file # noqa: INP001
"""Example script to test registration of frames of a video with VoxelMorph models."""

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
import tifffile as tiff
# from argparse import Namespace

# Third-party imports
import torch

# Local imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raft", "core"))
)  # noqa: PTH100, PTH118, PTH120
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voxelmorph_folder"
    )
)  # noqa: PTH100, PTH118, PTH120
from my_utils.configs import (  # type: ignore
    HEIGHT,
    TEST_DATA_GOWT,  # noqa: F401
    TEST_DATA_HELA,  # noqa: F401
    TEST_DATA_HSC,  # noqa: F401
    TEST_DATA_MUSC,  # noqa: F401
    TEST_SEG_DATA_GOWT,  # noqa: F401
    TEST_SEG_DATA_HELA,  # noqa: F401
    TEST_SEG_DATA_HSC,  # noqa: F401
    TEST_SEG_DATA_MUSC,  # noqa: F401
    WIDTH,
    device,
    MLFLOW_TRACKING_URI,
    SEED,
)
from my_utils.image_utils import interpole_and_pad, iou, viz, normalisation_params
from flow_visualizer import FlowVisualizer

# from raft import RAFT
from register_cells_old import (
    register_pair,
    register_pair_tvl1,
    register_pair_raft,
    register_pair_fb,
)
from model import WarperVxm

import voxelmorph as vxm

# for reproductability
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def evaluate_registration_with_seg(
    segmentation_moving, segmentation_fixed, displacement
):
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
    # Warp moving segmentation with the predicted displacement
    tensor_seg_moving = (
        torch.from_numpy(seg_moving).to(device).float().unsqueeze(0).unsqueeze(0)
    )
    spatial_transformer = vxm.nn.modules.SpatialTransformer(
        interpolation_mode="nearest"
    ).to(device)
    warped_seg_moving = spatial_transformer(tensor_seg_moving, displacement)
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
    ax[0, 0].imshow(seg_fixed, cmap="gray")
    plt.subplot(2, 2, 2)
    ax[0, 1].set_title("Warped moving segmentation")
    ax[0, 1].imshow(warped_seg_moving.squeeze().cpu().numpy(), cmap="gray")
    plt.subplot(2, 2, 3)
    ax[1, 0].set_title("Moving segmentation")
    ax[1, 0].imshow(seg_moving, cmap="gray")
    plt.subplot(2, 2, 4)
    ax[1, 1].set_title("Flow")
    viz(flo=displacement, ax=ax[1, 1])
    return score, fig


def evaluate_registration(moving, fixed, displacement):
    """Evaluate the registration results with MSE : not relevant (ideally find another method to evaluate the visual aspect)."""
    score = {}
    # Load segmentation masks
    moving = interpole_and_pad(moving, [HEIGHT, WIDTH])
    fixed = interpole_and_pad(fixed, [HEIGHT, WIDTH])
    # Warp moving with the predicted displacement
    tensor_moving = (
        torch.from_numpy(moving).to(device).float().unsqueeze(0).unsqueeze(0)
    )
    spatial_transformer = vxm.nn.modules.SpatialTransformer(
        interpolation_mode="linear"
    ).to(device)
    warped_moving = spatial_transformer(tensor_moving, displacement)
    # Compute MSE score
    mse = np.square(np.subtract(fixed, warped_moving.squeeze().cpu().numpy())).mean()
    ref_mse = np.square(
        np.subtract(fixed, moving)
    ).mean()  # reference dice to compare with the mse after registration
    score["mse"] = mse
    score["ref_mse"] = ref_mse
    return score


def evaluate_segmentation(segmentation_pred, segmentation_gt):
    """Evaluate the segmentation results."""
    score = {}
    # Load segmentation masks
    seg_pred = interpole_and_pad(segmentation_pred, [HEIGHT, WIDTH])
    seg_gt = interpole_and_pad(segmentation_gt, [HEIGHT, WIDTH])
    # Binarise
    seg_gt = seg_gt > 0
    seg_pred = (seg_pred > 0.5).astype(int)
    # Compute Dice score
    dice = vxm.py.utils.dice(seg_pred, seg_gt, include_zero=False)
    score["dice_segmentation"] = np.mean(dice)
    # To visulaise seg evaluation
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplot(1, 2, 1)
    ax[0].set_title("Segmentation prediction")
    ax[0].imshow(seg_pred, cmap="gray")
    plt.subplot(1, 2, 2)
    ax[1].set_title("Segmentation Ground Truth")
    ax[1].imshow(seg_gt, cmap="gray")
    return score, fig


def create_model(args):
    """Create the model for registration."""
    if args.model_type == "voxelmorph":
        # model = vxm.nn.models.VxmPairwise(
        #     ndim=2,
        #     source_channels=1,
        #     target_channels=1,
        #     nb_features=[16, 16, 16, 16, 16],
        #     integration_steps=0,
        #     device=device,
        # ).to(device) #TODO mettre option pour utiliser ce model vxm ?
        model = WarperVxm(
            ndim=2,
            source_channels=1,
            target_channels=1,
            nb_features=[16, 16, 16, 16, 16],
            model_type=args.model_type,
            device=device,
        ).to(device)
        model_path = f"{args.output_dir}/best.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Load model : {model_path}")  # noqa: T201
        model.eval()
        if mlflow.active_run():
            mlflow.log_param("model_path", model_path)
    elif args.model_type == "flowseg" or args.model_type == "twoflows":
        model = WarperVxm(
            ndim=2,
            source_channels=1,
            target_channels=1,
            nb_features=[16, 32, 64, 128, 256],
            model_type=args.model_type,
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
    # elif args.model_type == "raft":
    #     raft_args = Namespace(
    #         model="raft-things.pth",
    #         path=args.video_path,
    #         small=False,
    #         mixed_precision=False,
    #         alternate_corr=False,
    #     )
    #     model = torch.nn.DataParallel(RAFT(raft_args))
    #     model_path = "raft_models/raft-things.pth"
    #     model.load_state_dict(torch.load(model_path, map_location=device))

    #     model = model.module
    #     model.to(device)
    #     model.eval()
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

    # 1. Load video and segmentations frames
    frames_path_list = sorted(Path(args.video_path).iterdir())
    seg_path_list = (
        (sorted(Path(args.seg_path).iterdir())) if args.seg_path is not None else None
    )
    global_min, global_max = normalisation_params(frames_path_list)

    # MLFLOW
    if (
        args.model_type == "voxelmorph"
        or args.model_type == "twoflows"
        or args.model_type == "flowseg"
    ):
        with Path(f"{args.output_dir}/config.json").open() as j:
            config = json.load(j)  # To get informations from vxm training
    else:
        config = {}
        args.output_dir = f"outputs/{args.model_type}"

    if args.mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        parent_run_id = config.get("mlflow_run_id", None)
        mlflow.start_run(
            run_name=f"Test gap {args.gap} model type {args.model_type} on {args.video_path}",
            nested=True,
            parent_run_id=parent_run_id,  # To link the test run to the training run (if not specified, the test run will be a independant run)
        )
        print("MLflow tracking enabled.")  # noqa: T201
        mlflow.log_params(vars(args))

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
    for idx in range(min((len(frames_path_list) - args.gap), 100)):
        result = {}
        # Load moving and fixed images
        idx2 = idx + args.gap
        print(f"Registering pair : {idx} and {idx2}")  # noqa: T201
        # Read images
        moving = tiff.imread(frames_path_list[idx])
        fixed = tiff.imread(frames_path_list[idx2])
        # Interpole/Pad images to HEIGHT, WIDTH if needed
        moving = interpole_and_pad(moving, [HEIGHT, WIDTH])
        fixed = interpole_and_pad(fixed, [HEIGHT, WIDTH])
        # 3. Compute registration
        if (
            args.model_type == "voxelmorph"
            or args.model_type == "twoflows"
            or args.model_type == "flowseg"
        ):
            res = register_pair(moving, fixed, model, global_min, global_max)
            displacement, fig, input_moving, input_fixed = (
                res["displacement"],
                res["fig"],
                res["input_moving"],
                res["input_fixed"],
            )
            if "displacement_bwd" in res:
                displacement_bwd = res["displacement_bwd"]
            if "source_segmentation" in res:
                source_seg_pred = res["source_segmentation"]
        elif args.model_type == "tvl1":
            res = register_pair_tvl1(moving, fixed, global_min, global_max)
            displacement, fig, input_moving, input_fixed = (
                res["displacement"],
                res["fig"],
                res["input_moving"],
                res["input_fixed"],
            )
        elif args.model_type == "raft":
            res = register_pair_raft(
                frames_path_list[idx],
                frames_path_list[idx2],
                model,
                global_min,
                global_max,
            )
            displacement, fig, input_moving, input_fixed = (
                res["displacement"],
                res["fig"],
                res["input_moving"],
                res["input_fixed"],
            )
        elif args.model_type == "farnback":
            displacement, fig, input_moving, input_fixed = register_pair_fb(
                moving, fixed, model, global_min, global_max
            )
        fig.savefig(viz_dir / f"registration_result_for_paire_{idx}-{idx2}.png")
        plt.close(fig)

        if args.visualize or args.save_flow_maps:
            flow_maps[(idx2, idx)] = displacement.squeeze().cpu().numpy()
            if idx == 0:
                video.append(
                    np.expand_dims(input_moving, axis=-1)
                )  # channel dimension needed for visualization
            video.append(
                np.expand_dims(input_fixed, axis=-1)
            )  # channel dimension neededfor visualization
            if args.model_type == "voxelmorph" or args.model_type == "flowseg":
                res = register_pair(fixed, moving, model, global_min, global_max)
                displacement_bwd = res["displacement"]
            elif args.model_type == "tvl1":
                res = register_pair_tvl1(fixed, moving, global_min, global_max)
                displacement_bwd = res["displacement"]
            elif args.model_type == "raft":
                res = register_pair_raft(
                    frames_path_list[idx2],
                    frames_path_list[idx],
                    model,
                    global_min,
                    global_max,
                )
                displacement_bwd = res["displacement"]
            elif args.model_type == "farnback":
                displacement_bwd, _, _, _ = register_pair_fb(
                    fixed, moving, model, global_min, global_max
                )
            flow_maps[(idx, idx2)] = displacement_bwd.squeeze().cpu().numpy()

        if mlflow.active_run():
            mlflow.log_figure(
                fig, artifact_file=f"viz/registration_result_for_paire_{idx}-{idx2}.png"
            )

        # 4. Evaluation for the pair
        # Compute mse to evaluate visual aspect
        # score_pair = evaluate_registration(moving, fixed, displacement)
        # result.update(score_pair)
        # If segmentations available, load them and warp the moving segmentation with the predicted displacement for evaluation
        if args.seg_path is not None:
            # Load segmentation masks
            seg_moving = tiff.imread(seg_path_list[idx])
            seg_fixed = tiff.imread(seg_path_list[idx2])
            eval_res = evaluate_registration_with_seg(
                seg_moving, seg_fixed, displacement
            )
            if eval_res is None:
                continue
            score_pair, fig = eval_res
            result.update(score_pair)
            fig.savefig(viz_dir / f"seg_eval_for_paire_{idx}-{idx2}.png")
            if mlflow.active_run():
                mlflow.log_figure(
                    fig, artifact_file=f"seg_eval_for_paire_{idx}-{idx2}.png"
                )
            plt.close(fig)
            # ------------------------- SEGMENTATION EVALUATION ------------------------------------
            if "source_segmentation" in res:
                eval_seg_res = evaluate_segmentation(
                    source_seg_pred.squeeze().cpu().numpy(), seg_moving
                )
                if eval_seg_res is None:
                    continue
                score_pair, fig = eval_seg_res
                result.update(score_pair)
                fig.savefig(viz_dir / f"segmentation_result_for_{idx}.png")
                if mlflow.active_run():
                    mlflow.log_figure(
                        fig, artifact_file=f"segmentation_result_for_{idx}.png"
                    )
                plt.close(fig)
        else:
            print("Segmentation files not found, skipping segmentation evaluation.")  # noqa: T201
        results.append(result)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    metrics = {
        "data": args.video_path,
        "date": timestamp,
        "model": args.output_dir,
        "device": str(device),
        "gap": args.gap,
    }
    # 5. Global evaluation
    # Compute mse to evaluate the global visual aspect
    # mse_per_pair = [result["mse"] for result in results if result is not None]
    # ref_mse_scores_per_pair = [
    #     result["ref_mse"] for result in results if result is not None
    # ]
    # global_mse = np.mean(mse_per_pair)
    # global_ref_mse = np.mean(ref_mse_scores_per_pair)
    # metrics["global_mse"] = global_mse
    # metrics["global_ref_mse"] = global_ref_mse
    # Compute de global dice score an iou score
    if args.seg_path is not None:
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
        if "dice_segmentation" in results[0]:
            dice_segmentation_per_pair = [
                result["dice_segmentation"] for result in results if result is not None
            ]
            global_dice_segmentation = np.mean(dice_segmentation_per_pair)
        else:
            global_dice_segmentation = None
        # Save metrics in a json file
        metrics["global_dice"] = global_dice
        metrics["global_ref_dice"] = global_ref_dice
        metrics["global_iou"] = global_iou
        metrics["global_ref_iou"] = global_ref_iou
        metrics["dice_per_pair"] = dice_per_pair
        metrics["ref_dice_per_pair"] = ref_dice_scores_per_pair
        metrics["iou_per_pair"] = iou_scores_per_pair
        metrics["ref_iou_per_pair"] = ref_iou_scores_per_pair
        metrics["global_dice_segmentation"] = global_dice_segmentation
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

    if args.save_flow_maps:
        flow_dir = results_dir / "flowmaps"
        flow_dir.mkdir(parents=True, exist_ok=True)
        for name, value in flow_maps.items():
            np.save(flow_dir / f"{name}.npy", value)

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
        default=TEST_DATA_GOWT,
        help="Moving image (source) filename",
    )
    parser.add_argument("--gap", default=1, help="Temporal gap between 2 frames")
    parser.add_argument(
        "--seg_path",
        default=TEST_SEG_DATA_GOWT,
        help="Segmentations for evaluation (if segmentations not available, set to None)",
    )
    parser.add_argument(
        "--model_type",
        default="voxelmorph",
        choices=["voxelmorph", "twoflows", "flowseg", "tvl1", "raft", "farnback"],
        help="Type of model to use for registration",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/voxelmorph/output_20260720_172040",
        help="Directory to find model and to save results",
    )
    parser.add_argument(
        "--save_flow_maps",
        action="store_true",
        help="Save flow maps",
    )
    args = parser.parse_args()

    register_video(args)
