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
import numpy as np
import tifffile as tiff
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

# Local imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raft", "core"))
)  # noqa: PTH100, PTH118, PTH120
from my_utils.utils import interpole_and_pad
from my_utils.configs import (
    HEIGHT,
    TRAIN_DATA_MUSC,
    TRAIN_SEG_DATA_MUSC,
    WIDTH,
    NB_CLASSES,
    TRAIN_DATA,  # noqa: F401
    TRAIN_DATA_HSC,  # noqa: F401
    TRAIN_SEG_DATA,  # noqa: F401
    TRAIN_SEG_DATA_HSC,  # noqa: F401
    device,
)

import voxelmorph_folder.voxelmorph as vxm


class ThresholdTransform(nn.Module):
    """Return binarised object with a threshold and in the same type."""

    def __init__(self, thr_1):
        super().__init__()
        self.thr = thr_1

    def forward(self, x):
        """Forward."""
        return (x > self.thr).to(x.dtype)  # do not change the data type


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


class VxmVideoDataset(Dataset):
    """PyTorch Dataset for VoxelMorph registration data on video."""

    def __init__(
        self, video_path, seg_path=None, seg_type=None, max_gap=1, binarize=False
    ) -> None:
        """Parameters

        video_path : str
            Path to the video frames.
        seg_path : str
            Path to the segmentation frames (optional).
        max_gap : int
            Maximum gap between the two frames to select for registration (default: 20).
        seg_type : str
            Type of segmentation to use for training: None, labels or heatmaps.
        binarize : bool
        """
        self.video_path = Path(video_path)
        self.seg_path = Path(seg_path) if seg_path is not None else None
        self.seg_type = seg_type
        self.get_path_list()
        self.binarize = binarize
        self.transform = self.get_transforms(n=len(self.frames_path_list))
        if not max_gap < len(self.frames_path_list) // 2:
            raise ValueError(
                f"max_gap {max_gap} should be smaller than frames number // 2 {len(self.frames_path_list) // 2}."
            )
        self.max_gap = max_gap

    def __getitem__(self, idx):
        """Normalize video and generate pairs of frame from a single video.

        dict
            A dictionary containing the source and target frames.
        """
        # Select 2 frames with a random gap (maximum 10 frames gap)
        gap = np.random.randint(-self.max_gap, self.max_gap + 1)  # noqa: NPY002
        if gap == 0:
            gap = 1
        idx1 = idx % (
            len(self.frames_path_list) - 1
        )  # to avoid out of index error when selecting pairs of frames
        idx2 = (
            idx1 - gap
            if (idx1 + gap > (len(self.frames_path_list) - 1) or idx1 + gap < 0)
            else idx1 + gap
        )

        source_img = tiff.imread(self.frames_path_list[idx1])
        target_img = tiff.imread(self.frames_path_list[idx2])

        # Interpole and pad images to HEIGHT, WIDTH if needed
        moving_resized = interpole_and_pad(source_img, [HEIGHT, WIDTH])
        fixed_resized = interpole_and_pad(target_img, [HEIGHT, WIDTH])

        # To tensor
        source = torch.from_numpy(moving_resized).float()
        target = torch.from_numpy(fixed_resized).float()

        # Normalisation + augmentation if augment
        source = self.transform(source)
        target = self.transform(target)

        if self.seg_path is not None:
            source_seg = tiff.imread(self.seg_path_list[idx1])
            target_seg = tiff.imread(self.seg_path_list[idx2])
            source_seg = interpole_and_pad(source_seg, [HEIGHT, WIDTH])
            target_seg = interpole_and_pad(target_seg, [HEIGHT, WIDTH])
            if self.seg_type == "binary":
                source_seg = source_seg > 0
                target_seg = target_seg > 0
            elif self.seg_type == "labels":
                target_seg_oh = (  # noqa: F841
                    torch.nn.functional.one_hot(
                        target_seg.long(), num_classes=NB_CLASSES
                    )
                    .permute(0, 3, 1, 2)
                    .float()
                )
                source_seg_oh = (  # noqa: F841
                    torch.nn.functional.one_hot(
                        source_seg.long(), num_classes=NB_CLASSES
                    )
                    .permute(0, 3, 1, 2)
                    .float()
                )
                # Not enough memory to use labels (and one hot)
            source_seg = torch.from_numpy(source_seg).float()
            target_seg = torch.from_numpy(target_seg).float()
            return {
                "source": source,
                "target": target,
                "source_seg": source_seg,
                "target_seg": target_seg,
            }

        return {"source": source, "target": target}

    def get_transforms(self, n):
        """Get global min and global max from n frames and return normalisation and transformation functions."""
        global_min = np.min(tiff.imread(str(self.frames_path_list[0])))
        global_max = np.max(tiff.imread(str(self.frames_path_list[0])))

        for i in range(1, n):
            current_min = np.min(tiff.imread(str(self.frames_path_list[i])))
            current_max = np.max(tiff.imread(str(self.frames_path_list[i])))

            global_min = min(global_min, current_min)
            global_max = max(global_max, current_max)

        base_transforms = [
            NormalizeMinMax(global_min=global_min, global_max=global_max)
        ]

        if self.binarize:
            # base_transforms += [v2.RandomApply(v2.ColorJitter, 0.3)]  # noqa: ERA001
            base_transforms += [ThresholdTransform(thr_1=0.05)]

        return v2.Compose(base_transforms)

    def __len__(self):
        """Return nbr of paires to train on per epoch."""
        return (
            len(self.frames_path_list)
            * 1  # To have more paires per epoch TODO: add option to set this multiplier
        )

    def get_path_list(self):
        """Create a list with all the frame's path and segmentation's path."""
        self.frames_path_list = sorted(self.video_path.iterdir())
        if self.seg_path is not None:
            if self.seg_type == "heatmaps":
                self.seg_path = self.seg_path.parent / "SEG_HEATMAP"
            self.seg_path_list = sorted(self.seg_path.iterdir())


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    seg_loss_fn: nn.Module,
    loss_weights: Sequence[float],
    bidirectionnal: bool,
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
    bidirectionnal : bool
        Whether to use bidirectional training.
    """
    model.train()
    total_loss = 0.0
    epoch_img_loss = 0.0
    epoch_grad_loss = 0.0
    epoch_seg_loss = 0.0

    # Si get_iter dans dataset : for _ in range(steps_per_epoch): puis batch = next(dataloader)
    for batch in tqdm(dataloader, total=len(dataloader), desc="Training"):
        optimizer.zero_grad()

        # Move to device in training loop (not dataloader/dataset!)
        source = batch["source"]
        target = batch["target"]
        source = source.to(device).unsqueeze(1)  # Add channel dimension
        target = target.to(device).unsqueeze(1)  # Add channel dimension

        # Get the displacement and the warped source image from the model
        displacement, warped_source = model(
            source, target, return_warped_source=True, return_field_type="displacement"
        )

        # Compute losses
        img_loss = image_loss_fn(target, warped_source)
        if str(image_loss_fn) == "NCC()":
            img_loss = 1 - img_loss  # to minimize ncc loss
        grad_loss = grad_loss_fn(displacement)
        if str(grad_loss_fn) == "NCC()":
            grad_loss = 1 - grad_loss  # to minimize ncc loss

        loss = loss_weights[0] * img_loss + loss_weights[1] * grad_loss

        spatial_transformer = vxm.nn.modules.SpatialTransformer(
            interpolation_mode="linear"
        ).to(device)

        # If args.bidirectionnal, compute the cycle loss and add it to the total loss
        if bidirectionnal:
            displacement_inv, warped_target = model(
                target,
                source,
                return_warped_source=True,
                return_field_type="displacement",
            )

            half_dis = displacement * (0.5)
            half_dis_inv = displacement_inv * (0.5)
            half_warped_source = spatial_transformer(source, half_dis)
            half_warped_target = spatial_transformer(target, half_dis_inv)
            cycle_loss = image_loss_fn(half_warped_target, half_warped_source)
            if str(image_loss_fn) == "NCC()":
                cycle_loss = 1 - cycle_loss  # to minimize ncc loss

        # If segmentations are available, compute segmentation loss
        if "source_seg" in batch and "target_seg" in batch:
            source_seg = batch["source_seg"].to(device).unsqueeze(1)  # add channel dim
            target_seg = batch["target_seg"].to(device).unsqueeze(1)  # add channel dim
            warped_source_seg = spatial_transformer(source_seg, displacement)
            warped_source_seg = torch.clamp(warped_source_seg, 0, 1)
            # print(" NB CLASSES",len(torch.unique(target_seg.long()).tolist()),len(torch.unique(warped_source_seg).tolist()),)  # noqa: E501, ERA001
            seg_loss = seg_loss_fn(target_seg, warped_source_seg).squeeze()
            if str(seg_loss_fn) == "Dice()":
                seg_loss = 1 - seg_loss  # to minimize dice loss
            if background:
                # Add loss to avoid field in the background
                background_mask = (target_seg == 0).float()
                background_loss = torch.mean(
                    torch.abs(displacement) * background_mask
                )  # L1 loss on the displacement in the background
                loss += loss_weights[2] * background_loss  # TODO loss weight
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
    model = vxm.nn.models.VxmPairwise(
        ndim=2,
        source_channels=1,
        target_channels=1,
        nb_features=[16, 16, 16, 16, 16],
        integration_steps=0,
        device=device,
    ).to(device)

    # Setup losses and optimizer
    if args.image_loss == "mse":
        image_loss_fn = ne.nn.modules.MSE()
    elif args.image_loss == "cc":
        image_loss_fn = ne.nn.modules.NCC(window_size=4)
    else:
        raise ValueError(
            f"Unsupported image loss: {args.image_loss} use 'mse' or 'CC'."
        )
    grad_loss_fn = ne.nn.modules.SpatialGradient(args.grad_loss)
    if args.seg_type == "heatmaps":
        seg_loss_fn = (
            image_loss_fn  # use the same loss as for the image (MSE or CC) for heatmaps
        )
    else:
        seg_loss_fn = ne.nn.modules.Dice()
    loss_weights = [1.0, args.lambda_param, args.gamma_param]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create dataloader
    train_dataset = VxmVideoDataset(
        video_path=args.video_path,
        seg_path=args.seg_path,
        seg_type=args.seg_type,
        max_gap=args.max_gap,
        binarize=args.binarize,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,  # to speed up loading since data is not too big, set to False if not enough RAM
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"Training for {args.epochs} epochs...")  # noqa: T201
    best_loss = float("inf")
    # loss_during_training = []
    # img_loss_during_training = []
    # grad_loss_during_training = []
    # seg_loss_during_training = []
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        # Train for one epoch
        avg_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            seg_loss_fn=seg_loss_fn,
            loss_weights=loss_weights,
            bidirectionnal=args.bidirectionnal,
            background=args.background,
        )
        # Save loss for visualisation
        # loss_during_training.append(avg_loss["loss"])
        # img_loss_during_training.append(avg_loss["img_loss"])
        # grad_loss_during_training.append(avg_loss["grad_loss"])
        # if avg_loss["seg_loss"] is not None:
        #     seg_loss_during_training.append(avg_loss["seg_loss"])
        if mlflow.active_run():
            mlflow.log_metric("Train_loss", avg_loss["loss"], step=epoch)
            mlflow.log_metric("Train_img_loss", avg_loss["img_loss"], step=epoch)
            mlflow.log_metric("Train_grad_loss", avg_loss["grad_loss"], step=epoch)
            if avg_loss["seg_loss"] is not None:
                mlflow.log_metric("Train_seg_loss", avg_loss["seg_loss"], step=epoch)

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

    # # Loss visualisation
    # plt.plot(loss_during_training, label="Total Loss")
    # plt.plot(img_loss_during_training, label="Image Loss")
    # plt.plot(grad_loss_during_training, label="Gradient Loss")
    # if seg_loss_during_training:
    #     plt.plot(seg_loss_during_training, label="Segmentation Loss")
    # plt.title("Training loss")
    # plt.legend()
    # plt.savefig(output_dir / "loss.png")
    # Save final model
    final_path = output_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    config = {
        "model": "VxmPairwise",
        "max_gap": args.max_gap,
        "segmentations": args.seg_path is not None,
        "dataset": args.video_path,
        "binarization": args.binarize,
        "epochs": args.epochs,
        "image_loss": args.image_loss,
        "grad_loss": args.grad_loss,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lambda": args.lambda_param,
        "gamma": args.gamma_param,
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
    parser.add_argument(
        "--seg_type",
        type=str,
        default="binary",
        choices=["binary", "heatmaps", None],
        help="Type of segmentation to use for training: None, 'binary' or heatmaps",  # TODO add labels/tra?
    )
    parser.add_argument(
        "--binarize", type=bool, default=False, help="Binarize images for training"
    )
    parser.add_argument(
        "--max_gap", type=int, default=1, help="Max temporal gap between 2 frames"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument(
        "--image-loss", type=str, default="mse", help="Image loss to use: 'mse' or 'cc'"
    )
    parser.add_argument(
        "--grad_loss", type=str, default="l2", help="Gradient loss to use: 'l1' or 'l2'"
    )
    parser.add_argument("--workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--lambda",
        type=float,
        dest="lambda_param",
        default=0.1,
        help="Regularisation parameter for smooth loss",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        dest="gamma_param",
        default=0.001,
        help="Regularisation parameter for segmentation loss, not used if seg_path is set to None",
    )
    parser.add_argument(
        "--bidirectionnal", action="store_true", help="Train a bidirectional model"
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Add loss to avoid field in the background, only available if segmentations available",
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    parser.add_argument(
        "--save-every", type=int, default=10, help="Checkpoint every N epochs"
    )
    args = parser.parse_args()
    main(args)
