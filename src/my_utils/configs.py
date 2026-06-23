# noqa: INP001
# To run on gpu is available
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

# Seed for reproductability
SEED = 42

# Data Path
# GOWT video
TRAIN_DATA = "data/data/Fluo-N2DH-GOWT1_train/01"
TRAIN_SEG_DATA = "data/data/Fluo-N2DH-GOWT1_train/01_ERR_SEG_TRA"
NB_CLASSES = 36  # max number of cells in the video + 1 for the background
TEST_DATA = "data/data/Fluo-N2DH-GOWT1_train/02"
TEST_SEG_DATA = "data/data/Fluo-N2DH-GOWT1_train/02_ERR_SEG_TRA"
# HeLa video
TRAIN_DATA_HELA = "data/data/Fluo-N2DL-HeLa_train/01"
TRAIN_SEG_DATA_HELA = "data/data/Fluo-N2DL-HeLa_train/01_ERR_SEG_TRA"
TEST_DATA_HELA = "data/data/Fluo-N2DL-HeLa_train/02"
TEST_SEG_DATA_HELA = "data/data/Fluo-N2DL-HeLa_train/02_ERR_SEG_TRA"
# hsc video
TRAIN_DATA_HSC = "data/data/BF-C2DL-HSC/01"
TRAIN_SEG_DATA_HSC = "data/data/BF-C2DL-HSC/01_ERR_SEG_TRA"
TEST_DATA_HSC = "data/data/BF-C2DL-HSC/02"
TEST_SEG_DATA_HSC = "data/data/BF-C2DL-HSC/02_ERR_SEG_TRA"
# musc video
TRAIN_DATA_MUSC = "data/data/BF-C2DL-MuSC/01"
TRAIN_SEG_DATA_MUSC = "data/data/BF-C2DL-MuSC/01_ERR_SEG_TRA"
TEST_DATA_MUSC = "data/data/BF-C2DL-MuSC/02"
TEST_SEG_DATA_MUSC = "data/data/BF-C2DL-MuSC/02_ERR_SEG_TRA"


# Constants
HEIGHT = 1024
WIDTH = 1024
