#!/usr/bin/env python3
"""VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8).
pp 1788-1800. 2019.
"""  # noqa: D205

# Core library imports
import argparse
import json
import os
import sys
from collections.abc import Sequence  # noqa: TC003
from pathlib import Path

os.environ["NEURITE_BACKEND"] = "pytorch"
os.environ["VXM_BACKEND"] = "pytorch"
# Third-party imports
import datetime

import mlflow
import neurite as ne
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raft", "core"))
)  # noqa: PTH100, PTH118, PTH120
from src.dataset import VxmVideoDataset
from src.model import TwoHeadsVxm  # noqa: F401
from src.my_utils.configs import (
    TRAIN_DATA_MUSC,  # noqa: F401
    TRAIN_SEG_DATA_MUSC,  # noqa: F401
    TRAIN_DATA,  # noqa: F401
    TRAIN_DATA_HSC,  # noqa: F401
    TRAIN_SEG_DATA,  # noqa: F401
    TRAIN_SEG_DATA_HSC,  # noqa: F401
    device,
)
from src.my_utils.utils import linear_augment  # noqa: F401

import voxelmorph_folder.voxelmorph as vxm


class NormalizeMinMax(nn.Module):
    """Return normalized tensor based on min, max."""

    def __init__(self, global_min, global_max, eps=1e-7):
        super().__init__()
        self.min = global_min
        self.max = global_max
        self.eps = eps

    def forward(self, x):
        """Forward."""
        return (x - self.min) / (self.max - self.min + self.eps)


def pyramidal_flow(source, target, model, model_type, level=2):
    """Compute the pyramidal flow to allow the model to understand bigger movements."""
    current_field = None
    spatial_transformer = vxm.nn.modules.SpatialTransformer(
        interpolation_mode="linear"
    ).to(device)
    # Start from the lowest resolution and upsample the field at each level to use it as initial field for the next level
    for i in range(level):
        scale_factor = 1 / (2 ** (level - i - 1))  # 1/4, 1/2, 1 for 3 levels
        source_down = F.interpolate(
            source, scale_factor=scale_factor, mode="bilinear", align_corners=False
        )
        target_down = F.interpolate(
            target, scale_factor=scale_factor, mode="bilinear", align_corners=False
        )

        if current_field is None:
            if model_type == "voxelmorph":
                current_field = model(
                    source_down,
                    target_down,
                    return_warped_source=False,
                    return_field_type="displacement",
                )
            elif model_type == "twoheads_vxm":
                current_field, current_field_inv = model(
                    source_down,
                    target_down,
                    return_warped_source=False,
                )
        else:
            field_up = (
                F.interpolate(
                    current_field,
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=False,
                )
                * 2
            )  # because the field is in the original resolution, we need to multiply by 2 to get the correct field in the downsampled resolution
            warped_source_down = spatial_transformer(source_down, field_up)
            if model_type == "voxelmorph":
                field = model(
                    warped_source_down,
                    target_down,
                    return_warped_source=False,
                    return_field_type="displacement",
                )
            elif model_type == "twoheads_vxm":
                field, field_inv = model(
                    warped_source_down,
                    target_down,
                    return_warped_source=False,
                )
                field_inv_up = (
                    F.interpolate(
                        current_field_inv,
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=False,
                    )
                    * 2
                )
                current_field_inv = field_inv + field_inv_up
            current_field = field + field_up  # composition of the two fields
    if model_type == "voxelmorph":
        return field
    return field, field_inv


def train_epoch(
    epoch: int,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    seg_loss_fn: nn.Module,
    loss_weights: Sequence[float],
    background: bool,
) -> float:
    """Train for one epoch.

    Parameters
    ----------
    model : nn.Module
        The VoxelMorph model to train.
    dataloader : torch.utils.data.DataLoader
        The dataloader to use for training.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    image_loss_fn : nn.Module
        The image loss function to use.
    grad_loss_fn : nn.Module
        The gradient loss function to use.
    loss_weights : Sequence[float]
        The weights for the image and gradient losses.
    """
    model.train()
    total_loss = 0.0
    epoch_img_loss = 0.0
    epoch_grad_loss = 0.0
    epoch_seg_loss = 0.0
    epoch_background_loss = 0.0
    epoch_cycle_loss = 0.0

    # Si get_iter dans dataset : for _ in range(steps_per_epoch): puis batch = next(dataloader)
    for batch in tqdm(dataloader, total=len(dataloader), desc="Training"):
        optimizer.zero_grad()

        # Define warper for later use
        spatial_transformer = vxm.nn.modules.SpatialTransformer(
            interpolation_mode="linear"
        ).to(device)

        # Move to device in training loop (not dataloader/dataset!)
        source = batch["source"]
        target = batch["target"]
        source = source.to(device).unsqueeze(1)  # Add channel dimension
        target = target.to(device).unsqueeze(1)  # Add channel dimension
        if "source_seg" in batch and "target_seg" in batch:
            source_seg = batch["source_seg"].to(device).unsqueeze(1)  # add channel dim
            target_seg = batch["target_seg"].to(device).unsqueeze(1)  # add channel dim

        loss = 0.0
        img_loss = 0.0
        seg_loss = 0.0
        grad_loss = 0.0

        # Get the displacement and the warped source image from the model
        displacement, displacement_inv, warped_source = model(
            source,
            target,
            return_warped_source=True,
        )
        if args.pyramidal:
            displacement = pyramidal_flow(
                source, target, model, model_type="voxelmorph", level=2
            )
            warped_source = spatial_transformer(source, displacement)

        warped_target = spatial_transformer(target, displacement_inv)  # TODO
        # displacement_inv, warped_target = model(
        #     target, source, return_warped_source=True
        # )

        # Image loss for backward displacement
        if str(image_loss_fn) == "NCC()":
            img_loss += 1 - image_loss_fn(source, warped_target)  # to minimize ncc loss
        else:
            img_loss += image_loss_fn(source, warped_target)

        # Grad loss for backward displacement
        if str(grad_loss_fn) == "NCC()":
            grad_loss += 1 - grad_loss_fn(displacement_inv)  # to minimize ncc loss
        else:
            grad_loss += grad_loss_fn(displacement_inv)

        # Cycle loss
        warped_flow_inv = spatial_transformer(displacement_inv, displacement)
        cycle_loss = torch.norm(displacement + warped_flow_inv, p=2, dim=1).mean()
        # Add a progressiv weight to loss (start at 0, max at loss_weights[4])
        cycle_weight = linear_augment(
            start=5, end=15, max_weight=loss_weights[4], epoch=epoch
        )
        loss += cycle_weight * cycle_loss
        epoch_cycle_loss += cycle_loss.item()

        # Image loss for forward displacement
        if background:
            # Multiply image * mask to focus only on cells not background
            warped_source_seg = spatial_transformer(source_seg, displacement)
            warped_source_seg = torch.clamp(warped_source_seg, 0, 1)
            target = target * target_seg
            warped_source = warped_source * warped_source_seg
        if str(image_loss_fn) == "NCC()":
            img_loss += (
                1 - image_loss_fn(target, warped_source)[0][0]
            )  # to minimize ncc loss
        else:
            img_loss += image_loss_fn(target, warped_source)

        # Grad loss (smoothness)
        if str(grad_loss_fn) == "NCC()":
            grad_loss += 1 - grad_loss_fn(displacement)  # to minimize ncc loss
        else:
            grad_loss += grad_loss_fn(displacement)

        # Add to the total loss
        loss += loss_weights[0] * img_loss + loss_weights[1] * grad_loss

        # If segmentations are available, compute segmentation loss
        if "source_seg" in batch and "target_seg" in batch:
            seg_loss = 0.0
            warped_source_seg = spatial_transformer(source_seg, displacement)
            warped_source_seg = torch.clamp(warped_source_seg, 0, 1)
            # print(" NB CLASSES",len(torch.unique(target_seg.long()).tolist()),len(torch.unique(warped_source_seg).tolist()),)  # noqa: E501, ERA001
            if str(seg_loss_fn) == "Dice()":
                seg_loss += (
                    1 - seg_loss_fn(target_seg, warped_source_seg).squeeze()
                )  # to minimize dice loss
            else:
                seg_loss += seg_loss_fn(target_seg, warped_source_seg).squeeze()

            warped_target_seg = spatial_transformer(target_seg, displacement_inv)
            warped_target_seg = torch.clamp(warped_target_seg, 0, 1)
            # print(" NB CLASSES",len(torch.unique(target_seg.long()).tolist()),len(torch.unique(warped_source_seg).tolist()),)  # noqa: E501, ERA001
            if str(seg_loss_fn) == "Dice()":
                seg_loss += (
                    1 - seg_loss_fn(source_seg, warped_target_seg).squeeze()
                )  # to minimize dice loss
            else:
                seg_loss += seg_loss_fn(source_seg, warped_target_seg).squeeze()
            # Add loss to avoid field in the background
            if background:
                union_seg = torch.clamp(target_seg + source_seg, 0.0, 1.0)
                background_mask = 1.0 - union_seg
                background_loss = torch.mean(
                    torch.abs(displacement) * background_mask
                )  # L1 loss on the displacement in the background
                loss += loss_weights[3] * background_loss
                epoch_background_loss += background_loss.item()
            if args.seg_type is not None:
                loss += loss_weights[2] * seg_loss
                epoch_seg_loss += seg_loss.item()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        epoch_img_loss += img_loss.item()
        epoch_grad_loss += grad_loss.item()

    return {
        "loss": total_loss / len(dataloader),
        "img_loss": epoch_img_loss / len(dataloader),
        "grad_loss": epoch_grad_loss / len(dataloader),
        "seg_loss": epoch_seg_loss / len(dataloader) if epoch_seg_loss > 0 else None,
        "background_loss": epoch_background_loss / len(dataloader)
        if epoch_background_loss > 0
        else None,
        "cycle_loss": epoch_cycle_loss / len(dataloader)
        if epoch_cycle_loss > 0
        else None,
    }


def main(args):
    """Create the dataloader, create the model, train and save."""
    # Set device
    print(f"Using device: {device}")  # noqa: T201
    if args.seg_path is None:
        args.seg_type = None
        args.gamma_param = 0.0
    print(f"Using segmentation type: {args.seg_type}")  # noqa: T201
    # MLflow set up
    if args.mlflow:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.start_run(run_name=f"train_cells_{timestamp}_{args.video_path}")  # noqa: DTZ005
        print("MLflow tracking enabled.")  # noqa: T201
        mlflow.log_params(vars(args))

    # Create model
    model = TwoHeadsVxm(
        ndim=2,
        source_channels=1,
        target_channels=1,
        nb_features=[16, 16, 16, 16, 16],
        device=device,
    ).to(device)

    # Setup losses and optimizer
    if args.image_loss == "mse":
        image_loss_fn = ne.nn.modules.MSE()
    elif args.image_loss == "cc":
        image_loss_fn = ne.nn.modules.NCC(window_size=4)
    else:
        raise ValueError(
            f"Unsupported image loss: {args.image_loss} use 'mse' or 'cc'."
        )
    grad_loss_fn = ne.nn.modules.SpatialGradient(args.grad_loss)
    if args.seg_type == "heatmaps":
        seg_loss_fn = (
            image_loss_fn  # use the same loss as for the image (MSE or CC) for heatmaps
        )
    else:
        seg_loss_fn = ne.nn.modules.Dice()
    loss_weights = [
        1.0,
        args.lambda_param,
        args.gamma_param,
        args.background_param,
        args.cycle_param,
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create dataloader
    train_dataset = VxmVideoDataset(
        video_path=args.video_path,
        seg_path=args.seg_path,
        seg_type=args.seg_type,
        mode="train",
        gap=args.max_gap,
        multiplier=args.multiplier,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,  # To speed up, if not enough RAM, set to False
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"Training for {args.epochs} epochs...")  # noqa: T201
    best_loss = float("inf")
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        # Train for one epoch
        avg_loss = train_epoch(
            epoch=epoch,
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            seg_loss_fn=seg_loss_fn,
            loss_weights=loss_weights,
            background=args.background,
        )
        if mlflow.active_run():
            mlflow.log_metric("Train_loss", avg_loss["loss"], step=epoch)
            mlflow.log_metric("Train_img_loss", avg_loss["img_loss"], step=epoch)
            mlflow.log_metric(
                f"Train_grad_loss lambda:{args.lambda_param}",
                avg_loss["grad_loss"],
                step=epoch,
            )
            if avg_loss["seg_loss"] is not None:
                mlflow.log_metric(
                    f"Train_seg_loss gamma:{args.gamma_param}",
                    avg_loss["seg_loss"],
                    step=epoch,
                )
            if avg_loss["background_loss"] is not None:
                mlflow.log_metric(
                    f"Train_background_loss:{args.background_param}",
                    avg_loss["background_loss"],
                    step=epoch,
                )
            if avg_loss["cycle_loss"] is not None:
                mlflow.log_metric(
                    f"Train_cycle_loss:{args.cycle_param}",
                    avg_loss["cycle_loss"],
                    step=epoch,
                )
        # Print progress
        # if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss['loss']:.6f}")  # noqa: T201

        # Save periodic checkpoints
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoints/checkpoint_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")  # noqa: T201

        # Save best model
        if avg_loss["loss"] < best_loss:
            best_loss = avg_loss["loss"]
            best_path = output_dir / "best.pt"
            torch.save(model.state_dict(), best_path)
            if mlflow.active_run():
                mlflow.pytorch.log_model(model)

    # Save final model
    final_path = output_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    config = {
        "model": "VxmPairwise",
        "max_gap": args.max_gap,
        "segmentations": args.seg_path is not None,
        "dataset": args.video_path,
        "epochs": args.epochs,
        "image_loss": args.image_loss,
        "grad_loss": args.grad_loss,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lambda": args.lambda_param,
        "gamma": args.gamma_param,
        "background_param": args.background_param,
        "cycle_param": args.cycle_param,
        "device": str(device),
        "date": timestamp,
    }
    if mlflow.active_run():
        run_id = mlflow.active_run().info.run_id
        config["mlflow_run_id"] = run_id
    with Path.open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"Final model saved to {final_path}")  # noqa: T201

    # End MLflow run
    if mlflow.active_run():
        mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VoxelMorph")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    parser.add_argument(
        "--mlflow", action="store_true", help="Enable mlflow tracking during training"
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    # Directory and paths
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"outputs/output_{timestamp}",
        help="Output directory",
    )
    parser.add_argument(
        "--video-path", default=TRAIN_DATA_MUSC, help="Path to the training video"
    )
    parser.add_argument(
        "--seg-path",
        default=TRAIN_SEG_DATA_MUSC,
        help="Path to the segmentation of the training video, if not available set to None",
    )
    # Dataset parameters
    parser.add_argument(
        "--max_gap", type=int, default=1, help="Maximum temporal gap between 2 frames"
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=1,
        help="Multiplier to have more pairs per epoch",
    )
    parser.add_argument(
        "--seg_type",
        type=str,
        default="binary",
        choices=["binary", "heatmaps", None],
        help="Type of segmentation to use for training: None, 'binary' or heatmaps",  # TODO add labels/tra?
    )
    # Training parameters
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--save-every", type=int, default=10, help="Checkpoint every N epochs"
    )
    parser.add_argument(
        "--cycle_param",
        type=float,
        dest="cycle_param",
        default=0.005,
        help="Regularisation parameter for cycle loss",
    )
    parser.add_argument(
        "--image-loss", type=str, default="mse", help="Image loss to use: 'mse' or 'cc'"
    )
    parser.add_argument(
        "--grad_loss", type=str, default="l2", help="Gradient loss to use: 'l1' or 'l2'"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--lambda",
        type=float,
        dest="lambda_param",
        default=0.01,
        help="Regularisation parameter for smooth loss",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        dest="gamma_param",
        default=0.01,
        help="Regularisation parameter for segmentation loss, not used if seg_path is set to None",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Add loss to avoid field in the background, only available if segmentations available",
    )
    parser.add_argument(
        "--background_param",
        type=float,
        default=0.002,
        help="Parameter for background loss, only available if segmentations available",
    )
    parser.add_argument(
        "--pyramidal",
        action="store_true",
        help="Use pyramidal method for training, to allow the model to understand bigger movements",
    )
    args = parser.parse_args()
    main(args)
