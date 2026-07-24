# Core library imports
import argparse
import os

os.environ["NEURITE_BACKEND"] = "pytorch"
os.environ["VXM_BACKEND"] = "pytorch"

# Third-party imports
import numpy as np
import torch
import pandas as pd

# Local imports
# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raft", "core"))
# )  # noqa: PTH100, PTH118, PTH120
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# sys.path.append(
#     os.path.join(
#         os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voxelmorph_folder"
#     )
# )  # noqa: PTH100, PTH118, PTH120
from my_utils.configs import (
    TRAIN_DATA_MUSC,  # noqa: F401
    TRAIN_SEG_DATA_MUSC,  # noqa: F401
    TRAIN_DATA_GOWT,  # noqa: F401
    TRAIN_SEG_DATA_GOWT,  # noqa: F401
    TRAIN_DATA_HSC,  # noqa: F401
    TRAIN_SEG_DATA_HSC,  # noqa: F401
    TRAIN_DATA_HELA,  # noqa: F401
    TRAIN_SEG_DATA_HELA,  # noqa: F401
)
from train_cells import train
from tests import register_video


if __name__ == "__main__":
    seeds = [42, 123, 456, 54321, 98765]
    model = ["flowseg", "voxelmorph", "tvl1", "transflowseg"]
    datasets = ["GOWT", "HELA", "MUSC", "HSC"]

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
        "gamma_param": 0.01,
        "seg_param": 0.01,
        "background": False,
        "background_param": 0.01,
        "pyramidal": False,
        "multiple_videos": False,
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

    for model_name in model:
        dice_res_per_data = {}
        train_time_per_data = {}
        inf_time_per_data = {}
        for dataset in datasets:
            dice_res_per_seed = []
            train_time_per_seed = []
            inf_time_per_seed = []
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                video_path = f"TRAIN_DATA_{dataset}"
                seg_path = f"TRAIN_SEG_DATA_{dataset}"
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
                }
                train_args = argparse.Namespace(
                    **base_train_args, **additionnal_train_args
                )

                output_dir, train_time = train(train_args)

                additionnal_test_args = {
                    "video_path": f"TEST_DATA_{dataset}",
                    "seg_path": f"TEST_SEG_DATA_{dataset}",
                    "model_type": model_name,
                    "output_dir": output_dir,
                }

                test_args = argparse.Namespace(
                    **base_test_args, **additionnal_test_args
                )

                dice, inf_time = register_video(test_args)

                dice_res_per_seed.append(dice)
                train_time_per_seed.append(train_time)
                inf_time_per_seed.append(inf_time)

            dice_res_per_data[dataset] = (
                f"{str(np.mean(dice_res_per_seed))}({np.std(dice_res_per_seed)})"
            )
            train_time_per_data[dataset] = (
                f"{str(np.mean(train_time_per_seed))}({np.std(train_time_per_seed)})"
            )
            inf_time_per_data[dataset] = (
                f"{str(np.mean(inf_time_per_seed))}({np.std(inf_time_per_seed)})"
            )

        df = pd.DataFrame(data=dice_res_per_data)
        dice_df = pd.concat([dice_df, df], axis=0)

        df = pd.DataFrame(data=train_time_per_data)
        train_time_df = pd.concat([dice_df, df], axis=0)

        df = pd.DataFrame(data=inf_time_per_data)
        inf_time_df = pd.concat([inf_time_df, df], axis=0)

dice_df.to_csv("RESULTS.csv")
train_time_df.to_csv("TRAIN_TIME_RESULTS.csv")
inf_time_df.to_csv("INF_TIME_RESULTS.csv")
