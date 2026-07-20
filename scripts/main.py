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
import numpy as np

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
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voxelmorph_folder"
    )
)  # noqa: PTH100, PTH118, PTH120
from dataset import VxmVideoDataset
from model import WarperVxm  # noqa: F401
from my_utils.configs import (
    TRAIN_DATA_MUSC,  # noqa: F401
    TRAIN_SEG_DATA_MUSC,  # noqa: F401
    TRAIN_DATA,  # noqa: F401
    TRAIN_SEG_DATA,  # noqa: F401
    TRAIN_DATA_HSC,  # noqa: F401
    TRAIN_SEG_DATA_HSC,  # noqa: F401
    TRAIN_DATA_HELA,  # noqa: F401
    TRAIN_SEG_DATA_HELA,  # noqa: F401
    device,
    MLFLOW_TRACKING_URI,
    SEED,
)
from train_cells import main as train_main
from my_utils.image_utils import linear_augment  # noqa: F401

import voxelmorph as vxm

# for reproductability
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--video-path", default=TRAIN_DATA, help="Path to the training video"
    )
    parser.add_argument(
        "--seg-path",
        default=TRAIN_SEG_DATA,
        help="Path to the segmentation of the training video, if not available set to None",
    )
    # Dataset parameters
    parser.add_argument(
        "--max_gap", type=int, default=20, help="Maximum temporal gap between 2 frames"
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=3,
        help="Multiplier to have more pairs per epoch",
    )
    parser.add_argument(
        "--seg_type",
        type=str,
        default="binary",
        choices=["binary", "heatmaps", None],
        help="Type of segmentation to use for training: None, 'binary' or heatmaps",
    )
    # Training parameters
    parser.add_argument(
        "--model_type",
        type=str,
        default="voxelmorph",
        choices=["voxelmorph", "twoflows", "flowseg"],
        help="Type of model to train : voxelmorph, twoflows flowseg",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--save-every", type=int, default=10, help="Checkpoint every N epochs"
    )
    parser.add_argument(
        "--cycle_consistency",
        action="store_true",
        help="Add loss to enforce cycle consistency, automatic True if model_type is twoflows",
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
        help="Regularisation parameter for dice loss between source and warped target, not used if seg_path is set to None",
    )
    parser.add_argument(
        "--seg_param",
        type=float,
        dest="seg_param",
        default=0.01,
        help="Regularisation parameter for segmentation loss, not used if seg_path is set to None",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Add loss to avoid displacement in the background, only available if segmentations available",
    )
    parser.add_argument(
        "--background_param",
        type=float,
        default=0.01,
        help="Parameter for background loss, only available if segmentations available",
    )
    parser.add_argument(
        "--pyramidal",
        action="store_true",
        help="Use pyramidal method for training, to allow the model to understand bigger movements",
    )
    args = parser.parse_args()
    train_main(args)
