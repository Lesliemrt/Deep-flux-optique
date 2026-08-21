"""
Script to execute the complete pipeline : segmentation, training and testing different models with multiple seed and on multiple dataset.
"""

# Core library imports
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["NEURITE_BACKEND"] = "pytorch"
os.environ["VXM_BACKEND"] = "pytorch"

# Third-party imports
import datetime

import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from cellpose import models
from cellpose.io import imread

# Local imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raft", "core"))
)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voxelmorph_folder"
    )
)
from tests import register_video
from train_cells import train

from my_utils import configs

if __name__ == "__main__":
    seeds = [42, 123, 456, 54321, 98765]
    model = ["flowseg", "voxelmorph", "transflowseg", "transmorph2d", "tvl1"]
    datasets = ["MUSC"]

    base_train_args = {
        "mlflow": False,
        "workers": 4,
        "output_dir": "outputs",
        "seg_type": "binary",
        "epochs": 30,
        "batch_size": 4,
        "save_every": 15,
        "cycle_consistency": False,
        "cycle_param": 0.005,
        "image_loss": "mse",
        "grad_loss": "l2",
        "lr": 1e-3,
        "lambda_param": 0.01,
        "gamma_param": 0.001,
        "seg_param": 0.01,
        "background": False,
        "background_param": 0.01,
        "pyramidal": False,
    }

    base_test_args = {
        "mlflow": False,
        "visualize": False,
        "gap": 1,
        "save_flow_maps": True,
    }

    train_time_df = pd.DataFrame(columns=datasets)
    inf_time_df = pd.DataFrame(columns=datasets)
    dice_df = pd.DataFrame(columns=datasets)
    rmse_df = pd.DataFrame(columns=datasets)

    for model_name in model:
        dice_res_per_data = {}
        rmse_res_per_data = {}
        train_time_per_data = {}
        inf_time_per_data = {}
        for dataset in datasets:
            dice_res_per_seed = []
            rmse_res_per_seed = []
            train_time_per_seed = []
            inf_time_per_seed = []
            for seed in seeds:
                print(f"Using model {model_name}, dataset {dataset} and seed {seed}")
                torch.manual_seed(seed)
                np.random.seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                video_path = getattr(configs, f"TRAIN_DATA_{dataset}")
                seg_path = getattr(
                    configs, f"TRAIN_SEG_DATA_{dataset}"
                )  # to use given segmentation

                # ----------------------------------- SEGMENTATION ------------------------------------------
                # Create segmentation with Cellpose

                output_path = Path(video_path).parent / "CELLPOSE_SEG"

                if not output_path.exists():
                    # use Cellpose to create segmentations
                    output_path.mkdir(parents=True, exist_ok=True)
                    frames_path_list = sorted(Path(video_path).iterdir())
                    imgs = [imread(f) for f in frames_path_list]

                    model = models.CellposeModel(gpu=True)

                    masks, _, _ = model.eval(
                        imgs, flow_threshold=0.4, cellprob_threshold=0.0
                    )
                    for idx, mask in enumerate(masks):
                        mask_binary = (mask > 0).astype(np.float32)
                        tiff.imwrite(
                            output_path / f"cellpose_{idx:04d}.tif", mask_binary
                        )
                seg_path = output_path

                # ----------------------------------- TRAINING ------------------------------------------
                if dataset == "GOWT" or dataset == "HELA":
                    max_gap = 20
                    multiplier = 3
                else:
                    max_gap, multiplier = 1, 1

                additionnal_train_args = {
                    "video_path": video_path,
                    "seg_path": seg_path,
                    "model_type": model_name,
                    "max_gap": max_gap,
                    "multiplier": multiplier,
                    "multiple_videos": True,
                }

                train_args = argparse.Namespace(
                    **base_train_args, **additionnal_train_args
                )

                if model_name == "tvl1":
                    output_dir, train_time = None, 0
                else:
                    output_dir, train_time = train(train_args)

                # ----------------------------------- TESTING ------------------------------------------

                additionnal_test_args = {
                    "video_path": getattr(configs, f"TEST_DATA_{dataset}"),
                    "seg_path": getattr(configs, f"TEST_SEG_DATA_{dataset}"),
                    "model_type": model_name,
                    "output_dir": str(output_dir),
                }

                test_args = argparse.Namespace(
                    **base_test_args, **additionnal_test_args
                )

                dice, rmse, inf_time = register_video(test_args)

                dice_res_per_seed.append(dice)
                rmse_res_per_seed.append(rmse)
                train_time_per_seed.append(train_time)
                inf_time_per_seed.append(inf_time)

            dice_res_per_data[dataset] = (
                f"{np.mean(dice_res_per_seed)!s}({np.std(dice_res_per_seed)})"
            )
            train_time_per_data[dataset] = (
                f"{np.mean(train_time_per_seed)!s}({np.std(train_time_per_seed)})"
            )
            inf_time_per_data[dataset] = (
                f"{np.mean(inf_time_per_seed)!s}({np.std(inf_time_per_seed)})"
            )
            rmse_res_per_data[dataset] = (
                f"{np.mean(rmse_res_per_seed)!s}({np.std(rmse_res_per_seed)})"
            )

        df = pd.DataFrame(data=dice_res_per_data, index=[model_name])
        dice_df = pd.concat([dice_df, df], axis=0)

        df = pd.DataFrame(data=train_time_per_data, index=[model_name])
        train_time_df = pd.concat([train_time_df, df], axis=0)

        df = pd.DataFrame(data=rmse_res_per_data, index=[model_name])
        rmse_df = pd.concat([rmse_df, df], axis=0)

        df = pd.DataFrame(data=inf_time_per_data, index=[model_name])
        inf_time_df = pd.concat([inf_time_df, df], axis=0)

timestamp = datetime.datetime.now(tz=datetime.UTC)
res_path = Path(f"RESULTS_CP_less_seg_{timestamp.strftime('%Y%m%d_%H%M%S')}")
res_path.mkdir(parents=True, exist_ok=True)
with Path.open(
    res_path / f"training_args.json_{timestamp.strftime('%Y%m%d_%H%M%S')}", "w"
) as f:
    json.dump(base_train_args, f, indent=4)
dice_df.to_csv(res_path / f"RESULTS_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv")
train_time_df.to_csv(
    res_path / f"TRAIN_TIME_RESULTS_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
)
inf_time_df.to_csv(
    res_path / f"INF_TIME_RESULTS_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
)
rmse_df.to_csv(res_path / f"RMSE_RESULTS_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv")
