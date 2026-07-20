# Core library imports
import argparse
from html import parser
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

import neurite as ne
import torch

import torch.nn.functional as F


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
    seeds = [42, 123, 456, 54321, 98765]
    model = ["flowseg", "voxelmorph"]
    datasets = ["GOWT", "HELA", "MUSC", "HSC"]

    base_args = {
        "workers": 4,
        "output_dir": "outputs",
        "seg_type": "binary",
        "model_type": "voxelmorph",
        "epochs": 30,
        "batch_size": 4,
        "save_every": 15,
        "cycle_consistency": False,
        "cycle_param": 0.005,
        "image_loss": "mse",
        "grad_loss": "l2",
        "lr": 1e-3,
        "lambda_param": 0.01,
        "gamma_param": 0.01,
        "seg_param": 0.01,
        "background": False,
        "background_param": 0.01,
        "pyramidal": False,
        "multiple_videos": False,
    }

    for seed in seeds:
        for model_name in model:
            for dataset in datasets:
                video_path = f"TRAIN_DATA_{dataset}"
                seg_path = f"TRAIN_SEG_DATA_{dataset}"
                if dataset == "GOWT" or dataset == "HELA":
                    max_gap = 20
                    multiplier = 3
                else:
                    max_gap, multiplier = 1, 1
                other_args = {
                    "video_path": video_path,
                    "seg_path": seg_path,
                    "max_gap": max_gap,
                    "multiplier": multiplier,
                }
                args = argparse.Namespace(**base_args, **other_args)

                train_main(args)

                # il faudrait que train return final_path
