# Core library imports
import os
import sys
from collections.abc import Sequence  # noqa: TC003


os.environ["NEURITE_BACKEND"] = "pytorch"
os.environ["VXM_BACKEND"] = "pytorch"
# Third-party imports
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


class Trainer:
    def __init__(self):
        # Define warper for later use
        self.spatial_transformer = vxm.nn.modules.SpatialTransformer(
            interpolation_mode="linear"
        ).to(device)

    def train_epoch(
        epoch: int,
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
        epoch_background_loss = 0.0
        epoch_cycle_loss = 0.0

        # Si get_iter dans dataset : for _ in range(steps_per_epoch): puis batch = next(dataloader)
        for batch in tqdm(dataloader, total=len(dataloader), desc="Training"):
            optimizer.zero_grad()

            # Move to device in training loop (not dataloader/dataset!)
            source = batch["source"]
            target = batch["target"]
            source = source.to(device).unsqueeze(1)  # Add channel dimension
            target = target.to(device).unsqueeze(1)  # Add channel dimension
            if "source_seg" in batch and "target_seg" in batch:
                source_seg = (
                    batch["source_seg"].to(device).unsqueeze(1)
                )  # add channel dim
                target_seg = (
                    batch["target_seg"].to(device).unsqueeze(1)
                )  # add channel dim

            loss = 0.0
            img_loss = 0.0
            seg_loss = 0.0
            grad_loss = 0.0

            # Get the displacement and the warped source image from the model
            # displacement, warped_source = model(
            #     source,
            #     target,
            #     return_warped_source=True,
            #     return_field_type="displacement",
            # )  # TODO
            if args.pyramidal:
                displacement = pyramidal_flow(
                    source, target, model, model_type="voxelmorph", level=2
                )
                warped_source = spatial_transformer(source, displacement)
            if bidirectionnal:
                displacement, displacement_inv, warped_source = model(
                    source, target, return_warped_source=True
                )
                warped_target = spatial_transformer(target, displacement_inv)  # TODO
                # displacement_inv, warped_target = model(
                #     target, source, return_warped_source=True
                # )

                # Image loss for backward displacement
                if str(image_loss_fn) == "NCC()":
                    img_loss += 1 - image_loss_fn(
                        source, warped_target
                    )  # to minimize ncc loss
                else:
                    img_loss += image_loss_fn(source, warped_target)

                # Grad loss for backward displacement
                if str(grad_loss_fn) == "NCC()":
                    grad_loss += 1 - grad_loss_fn(
                        displacement_inv
                    )  # to minimize ncc loss
                else:
                    grad_loss += grad_loss_fn(displacement_inv)

                # Cycle loss
                warped_flow_inv = spatial_transformer(displacement_inv, displacement)
                cycle_loss = torch.norm(
                    displacement + warped_flow_inv, p=2, dim=1
                ).mean()
                # Add a progressiv weight to loss (start at 0, max at loss_weights[4])
                cycle_weight = linear_augment(
                    start=5, end=15, max_weight=loss_weights[4], epoch=epoch
                )
                loss += cycle_weight * cycle_loss
                epoch_cycle_loss += cycle_loss.item()
            else:
                displacement, warped_source = model(
                    source,
                    target,
                    return_warped_source=True,
                    return_field_type="displacement",
                )  # TODO

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

            # If args.bidirectionnal, compute the cycle loss and add it to the total loss
            # if bidirectionnal:
            #     displacement_inv, warped_target = model(
            #         target,
            #         source,
            #         return_warped_source=True,
            #         return_field_type="displacement",
            #     )

            #     half_dis = displacement * (0.5)
            #     half_dis_inv = displacement_inv * (0.5)
            #     half_warped_source = spatial_transformer(source, half_dis)
            #     half_warped_target = spatial_transformer(target, half_dis_inv)
            #     cycle_loss = image_loss_fn(half_warped_target, half_warped_source)
            #     if str(image_loss_fn) == "NCC()":
            #         cycle_loss = 1 - cycle_loss  # to minimize ncc loss
            #     loss += loss_weights[0] * cycle_loss
            #     epoch_cycle_loss += cycle_loss.item()
            # loss += loss_weights[0] * img_loss

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
                if bidirectionnal:
                    warped_target_seg = spatial_transformer(
                        target_seg, displacement_inv
                    )
                    warped_target_seg = torch.clamp(warped_target_seg, 0, 1)
                    # print(" NB CLASSES",len(torch.unique(target_seg.long()).tolist()),len(torch.unique(warped_source_seg).tolist()),)  # noqa: E501, ERA001
                    if str(seg_loss_fn) == "Dice()":
                        seg_loss += (
                            1 - seg_loss_fn(source_seg, warped_target_seg).squeeze()
                        )  # to minimize dice loss
                    else:
                        seg_loss += seg_loss_fn(source_seg, warped_target_seg).squeeze()
                # if background:
                #     # Add loss to avoid field in the background
                #     union_seg = torch.clamp(target_seg + source_seg, 0.0, 1.0)
                #     background_mask = 1.0 - union_seg
                #     background_loss = torch.mean(
                #         torch.abs(displacement) * background_mask
                #     )  # L1 loss on the displacement in the background
                #     loss += loss_weights[3] * background_loss
                #     epoch_background_loss += background_loss.item()
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
            "seg_loss": epoch_seg_loss / len(dataloader)
            if epoch_seg_loss > 0
            else None,
            "background_loss": epoch_background_loss / len(dataloader)
            if epoch_background_loss > 0
            else None,
            "cycle_loss": epoch_cycle_loss / len(dataloader)
            if epoch_cycle_loss > 0
            else None,
        }
