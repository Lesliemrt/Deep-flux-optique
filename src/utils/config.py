# noqa: INP001
# To run on gpu is available
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
