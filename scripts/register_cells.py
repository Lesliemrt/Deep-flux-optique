#!/usr/bin/env python
# noqa: EXE001
"""Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca.
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019.

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu.
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under
the License.
"""

# Core library imports
import argparse

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

# Third-party imports
import torch

# Local imports
import voxelmorph as vxm  # nopep8
from utils.configs import DATA, device


def create_grid(shape, spacing=20):
    """Create a grid image with lines every `spacing` pixels."""
    grid = np.zeros(shape)
    grid[::spacing, :] = 1
    grid[:, ::spacing] = 1
    return grid


def register(args):
    """Register two volumes with a VoxelMorph model.

    Parameters
    ----------
    args : argparse.Namespace
        Commandline arguments.
    """
    # Load moving and fixed images
    moving = tiff.imread(args.moving)
    fixed = tiff.imread(args.fixed)

    # Normalize
    global_min = min(np.min(moving), np.min(fixed))
    global_max = max(np.max(moving), np.max(fixed))

    moving = (moving - global_min) / (global_max - global_min)
    fixed = (fixed - global_min) / (global_max - global_min)

    # load and set up model
    # Set device
    print(f"Using device: {device}")  # noqa: T201

    # Create model
    model = vxm.nn.models.VxmPairwise(
        ndim=2,
        source_channels=1,
        target_channels=1,
        nb_features=[16, 16, 16, 16, 16],
        integration_steps=0,
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # set up tensors
    input_moving = torch.from_numpy(moving).to(device).float().unsqueeze(0).unsqueeze(0)
    input_fixed = torch.from_numpy(fixed).to(device).float().unsqueeze(0).unsqueeze(0)

    # predict
    with torch.no_grad():
        field, moved = model(
            source=input_moving, target=input_fixed, return_warped_source=True, return_field_type="displacement"
        )

    grid = create_grid(moving.shape)
    grid = torch.from_numpy(grid).to(device).float().unsqueeze(0).unsqueeze(0)
    warped_grid = model.spatial_transformer(grid, field)

    return {"fixed": fixed, "warped_grid": warped_grid, "moved": moved}


if __name__ == "__main__":
    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving", default=f"{DATA}/01/t090.tif", help="moving image (source) filename")
    parser.add_argument("--fixed", default=f"{DATA}/01/t091.tif", help="fixed image (target) filename")
    parser.add_argument("--model", default="output/final.pt", help="weights of trained model to use for registration")
    # TODO : case 3D ?
    args = parser.parse_args()
    result = register(args)
    warped_grid = result["warped_grid"]
    moved = result["moved"]
    fixed = result["fixed"]
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.title("Fixed")
    plt.imshow(fixed, cmap="gray")
    plt.subplot(2, 2, 2)
    plt.title("Moved")
    plt.imshow(moved.squeeze().cpu().numpy(), cmap="gray")
    plt.subplot(2, 2, 3)
    plt.title("Difference (Fixed - Moved)")
    plt.imshow(fixed - moved.squeeze().cpu().numpy(), cmap="gray")
    plt.subplot(2, 2, 4)
    plt.title("Warped Grid")
    plt.imshow(warped_grid.squeeze().cpu().numpy(), cmap="gray")
    plt.show()
    # TODO save fig ?
