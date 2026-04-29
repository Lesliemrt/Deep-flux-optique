# noqa: INP001
# To run on gpu is available
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Path
DATA = "data/train_data/Fluo-N2DH-GOWT1_train"  # TODO more generic path
DATA2 = "data/train_data/Fluo-N2DL-HeLa_train"

# Constants
HEIGHT = 1024
WIDTH = 1024
