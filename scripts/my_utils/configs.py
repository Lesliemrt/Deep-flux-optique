# noqa: INP001
# To run on gpu is available
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed for reproductability
SEED = 42

# Data Path
# GOWT video
TRAIN_DATA = "data/train_data/Fluo-N2DH-GOWT1_train/01"
TRAIN_SEG_DATA = "data/train_data/Fluo-N2DH-GOWT1_train/01_ERR_SEG_TRA"
NB_CLASSES = 36  # max number of cells in the video + 1 for the background
TEST_DATA = "data/train_data/Fluo-N2DH-GOWT1_train/02"
TEST_SEG_DATA = "data/train_data/Fluo-N2DH-GOWT1_train/02_ERR_SEG_TRA"
# HeLa video
TRAIN_DATA_2 = "data/train_data/Fluo-N2DL-HeLa_train/01"
TRAIN_SEG_DATA_2 = "data/train_data/Fluo-N2DL-HeLa_train/01_ERR_SEG_TRA"
TEST_DATA_2 = "data/train_data/Fluo-N2DL-HeLa_train/02"
TEST_SEG_DATA_2 = "data/train_data/Fluo-N2DL-HeLa_train/02_ERR_SEG_TRA"

# Constants
HEIGHT = 1024
WIDTH = 1024
