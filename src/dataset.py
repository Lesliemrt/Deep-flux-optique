# Core library imports
import os
import sys
from pathlib import Path

os.environ["NEURITE_BACKEND"] = "pytorch"
os.environ["VXM_BACKEND"] = "pytorch"
# Third-party imports
import numpy as np
import tifffile as tiff
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import v2

# Local imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raft", "core"))
)  # noqa: PTH100, PTH118, PTH120
from my_utils.image_utils import interpole_and_pad
from my_utils.configs import (
    HEIGHT,
    TRAIN_DATA_MUSC,  # noqa: F401
    TRAIN_SEG_DATA_MUSC,  # noqa: F401
    WIDTH,
    NB_CLASSES,
    TRAIN_DATA,  # noqa: F401
    TRAIN_DATA_HSC,  # noqa: F401
    TRAIN_SEG_DATA,  # noqa: F401
    TRAIN_SEG_DATA_HSC,  # noqa: F401
)

# import voxelmorph_folder.voxelmorph as vxm


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
        self,
        video_path,
        seg_path=None,
        mode="train",
        seg_type=None,
        gap=1,
        multiplier=3,
    ) -> None:
        """Parameters

        video_path : str
            Path to the video frames.
        seg_path : str
            Path to the segmentation frames (optional).
        mode : str
            Mode of the dataset: "train" or "test".
        seg_type : str
            Type of segmentation to use for training: None, "binary", "labels" or "heatmaps".
        gap : int
            In training mode : Maximum gap between the two frames to select for registration, in test mode : fixed gap between the two frames to select for registration.
        seg_type : str
            Type of segmentation to use for training: None, labels or heatmaps.
        """
        self.video_path = Path(video_path)
        self.seg_path = Path(seg_path) if seg_path is not None else None
        self.mode = mode
        self.seg_type = seg_type
        self.get_path_list()
        self.transform = self.get_transforms(n=len(self.frames_path_list))
        if not gap < len(self.frames_path_list) // 2:
            raise ValueError(
                f"max_gap {gap} should be smaller than frames number // 2 {len(self.frames_path_list) // 2}."
            )
        self.gap = gap
        self.multiplier = multiplier

    def __getitem__(self, idx):
        """Normalize video and generate pairs of frame from a single video.

        dict
            A dictionary containing the source and target frames.
        """
        if self.mode == "train":
            # Select 2 frames with a random gap (maximum 10 frames gap)
            gap = np.random.randint(-self.gap, self.gap + 1)
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
        if self.mode == "test":
            # Select 2 frames with a fixed gap
            idx1 = idx
            idx2 = idx1 + self.gap

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
            # For testing, we don't need further processing
            if self.mode == "train":
                if self.seg_type == "binary" or self.seg_type is None:
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
                "idx1": idx1,
                "idx2": idx2,
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

        return v2.Compose(base_transforms)

    def __len__(self):
        """Return nbr of paires to train on per epoch."""
        if self.mode == "train":
            return len(self.frames_path_list) * self.multiplier
        else:
            return len(self.frames_path_list) - self.gap

    def get_path_list(self):
        """Create a list with all the frame's path and segmentation's path."""
        self.frames_path_list = sorted(self.video_path.iterdir())
        if self.seg_path is not None:
            if self.seg_type == "heatmaps":
                self.seg_path = self.seg_path.parent / "SEG_HEATMAP"
            self.seg_path_list = sorted(self.seg_path.iterdir())
