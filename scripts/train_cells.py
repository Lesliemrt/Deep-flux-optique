#!/usr/bin/env python3
"""VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8).
pp 1788-1800. 2019.
"""  # noqa: D205

# Core library imports
import argparse
import os
import sys
from collections.abc import Sequence  # noqa: TC003
from pathlib import Path

import neurite as ne

# Third-party imports
import numpy as np
import tifffile as tiff
import torch
from torch import nn  # noqa: TC002
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa: PTH100, PTH118, PTH120
from utils.configs import DATA, device

import voxelmorph_folder.voxelmorph as vxm


class VxmVideoDataset(Dataset):
    """PyTorch Dataset for VoxelMorph registration data on video."""

    def __init__(self, augment=False) -> None:
        """Parameters

        augment : bool
            Apply augmentations if True.
        """
        self.video_path = Path(DATA + "/01")
        self.get_frames_path_list()
        self.augment = augment
        self.transform = self.get_transforms(n=len(self))

    def __getitem__(self, idx):  # TODO: get item ou get_iter ?
        """Normalize video and generate pairs of frame from a single video.

        dict
            A dictionary containing the source and target frames.
        """
        # Select 2 frames
        idx1 = idx
        idx2 = idx + 1  # TODO 1 à mettre en arg pour entrainer sur n frame d'écart ?

        source_img = tiff.imread(self.frames_path_list[idx1])
        target_img = tiff.imread(self.frames_path_list[idx2])

        # Normalisation + augmentation if augment
        source_img = self.transform(source_img)
        target_img = self.transform(target_img)

        # To tensor
        source = torch.from_numpy(source_img)
        target = torch.from_numpy(target_img)

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

        # self.global_min = global_min  # noqa: ERA001
        # self.global_max = global_max  # noqa: ERA001

        def normalise(frame):
            return (frame - global_min) / (global_max - global_min)

        base_transforms = [normalise]

        if self.augment:
            base_transforms += [transforms.RandomApply(transforms.ColorJitter, 0.3)]

        return transforms.Compose(base_transforms)

    def __len__(self):
        """Return len dataset."""
        return len(self.frames_path_list) - 1  # to avoid out of index error when selecting pairs of frames

    def get_frames_path_list(self):
        """Create a list with all the frame's path."""
        self.frames_path_list = list(self.video_path.iterdir())


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    loss_weights: Sequence[float],
    # steps_per_epoch: int,
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
    steps_per_epoch : int
    """
    model.train()
    total_loss = 0.0

    # Si get_iter dans dataset : for _ in range(steps_per_epoch): puis batch = next(dataloader)
    for batch in tqdm(dataloader, total=len(dataloader), desc="Training"):
        optimizer.zero_grad()

        # Move to device in training loop (not dataloader/dataset!)
        source = batch["source"]
        target = batch["target"]
        source = source.float().to(device).unsqueeze(1)  # Add channel dimension
        target = target.float().to(device).unsqueeze(1)  # Add channel dimension

        # Get the displacement and the warped source image from the model
        displacement, warped_source = model(source, target, return_warped_source=True, return_field_type="displacement")

        img_loss = image_loss_fn(target, warped_source)
        grad_loss = grad_loss_fn(displacement)

        loss = loss_weights[0] * img_loss + loss_weights[1] * grad_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    """Create the dataloader, create the model, train and save."""
    parser = argparse.ArgumentParser(description="Train VoxelMorph")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--steps-per-epoch", type=int, default=100, help="Steps per epoch")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda", type=float, dest="lambda_param", default=0.01)
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint every N epochs")
    args = parser.parse_args()

    # Set device
    print(f"Using device: {device}")  # noqa: T201

    # Create model
    model = vxm.nn.models.VxmPairwise(
        ndim=2,
        source_channels=1,
        target_channels=1,
        nb_features=[16, 16, 16, 16, 16],
        integration_steps=0,
    ).to(device)

    # Setup losses and optimizer
    image_loss_fn = ne.nn.modules.MSE()
    grad_loss_fn = ne.nn.modules.SpatialGradient("l2")
    loss_weights = [1.0, args.lambda_param]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create dataloader
    train_dataset = VxmVideoDataset()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"Training for {args.epochs} epochs...")  # noqa: T201
    best_loss = float("inf")
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        # Train for one epoch
        avg_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            loss_weights=loss_weights,
        )

        # Print progress
        # if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}")  # noqa: T201

        # Save periodic checkpoints
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoints/checkpoint_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")  # noqa: T201

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_dir / "best.pt"
            torch.save(model.state_dict(), best_path)

    # Save final model
    final_path = output_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")  # noqa: T201


if __name__ == "__main__":
    main()
