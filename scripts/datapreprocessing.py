# To do once on segmentation dataset
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile as tiff
from cellpose import models
from cellpose.io import imread
from skimage.measure import regionprops
from tqdm import tqdm


def relabel_cells(array, with_labels):
    """Relabel an segmentation mask with labels from another one."""
    relabeled = np.zeros_like(array)
    for region in regionprops(array):
        if region.label == 0:
            continue
        # Get the most common label in the region from with_labels
        labels_in_region = with_labels[region.coords[:, 0], region.coords[:, 1]]
        label_count = Counter(labels_in_region)
        if max(label_count.keys()) == 0:
            most_common_label = 0
        else:
            label_count.pop(0, None)
            most_common_label = label_count.most_common(1)[0][0]
        relabeled[region.coords[:, 0], region.coords[:, 1]] = most_common_label
    return relabeled


def relabel_video(seg_video_path, tra_video_path, output_path):
    """Relabel a segmentation video with labels from a tra video."""
    err_seg_path_list = sorted(
        f for f in seg_video_path.iterdir() if f.suffix == ".tif"
    )
    tra_path_list = sorted(f for f in tra_video_path.iterdir() if f.suffix == ".tif")
    for idx in tqdm(range(len(err_seg_path_list)), desc="Relabeling video..."):
        err_seg = tiff.imread(err_seg_path_list[idx])
        tra = tiff.imread(tra_path_list[idx])
        relabeled = relabel_cells(err_seg, tra)
        tiff.imwrite(output_path / f"relabeled_{idx:04d}.tif", relabeled)


def mask_to_gaussian(segmentation, sigma_scale=0.25):
    """Convert a segmentation mask to a Gaussian heatmap.

    Parameters:
        segmentation: 2D array of shape (H, W) with integer labels for each cell.
        sigma_scale: Sigma for the Gaussian = radius * sigma_scale."""
    heatmap = np.zeros_like(segmentation, dtype=np.float32)
    y_grid, x_grid = np.indices(segmentation.shape)
    for region in regionprops(segmentation):
        if region.label == 0:
            continue
        y, x = region.centroid
        cell_sigma = (region.equivalent_diameter_area / 2) * sigma_scale
        cell_sigma = max(
            cell_sigma, 1
        )  # Ensure sigma is at least 1 to avoid very sharp peaks
        heatmap += np.exp(
            -((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * cell_sigma**2)
        ).astype(np.float32)
    return heatmap


def cellpose(video_path):

    output_path = Path(video_path).parent / "CELLPOSE_SEG"

    if output_path.exists():
        raise ValueError(f"Output path : {output_path} already exists")
    else:
        # use Cellpose to create segmentations
        output_path.mkdir(parents=True, exist_ok=True)
        frames_path_list = sorted(Path(video_path).iterdir())
        imgs = [imread(f) for f in frames_path_list]

        model = models.CellposeModel(gpu=True)

        masks, _, _ = model.eval(imgs, flow_threshold=0.4, cellprob_threshold=0.0)
        for idx, mask in enumerate(masks):
            mask_binary = (mask > 0).astype(np.float32)
            tiff.imwrite(output_path / f"cellpose_{idx:04d}.tif", mask_binary)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VoxelMorph")
    parser.add_argument(
        "--path",
        default="PhC-C2DH-U373",
        help="Path to the dataset to process",
    )
    args = parser.parse_args()

    # Create segmentations
    _ = cellpose("data/data/Fluo-C2DL-Huh7/01")

    # # Relabel the segmentation videos with the labels from the tracking videos
    # output_path = Path(f"data/data/{args.path}/01_ERR_SEG_TRA")
    # output_path.mkdir(exist_ok=True)
    # relabel_video(
    #     seg_video_path=Path(f"data/data/{args.path}/01_ERR_SEG"),
    #     tra_video_path=Path(f"data/data/{args.path}/01_GT/TRA"),
    #     output_path=output_path,
    # )
    # output_path = Path(f"data/data/{args.path}/02_ERR_SEG_TRA")
    # output_path.mkdir(exist_ok=True)
    # relabel_video(
    #     seg_video_path=Path(f"data/data/{args.path}/02_ERR_SEG"),
    #     tra_video_path=Path(f"data/data/{args.path}/02_GT/TRA"),
    #     output_path=output_path,
    # )

    # # Create Gaussian heatmaps from the relabeled segmentation videos
    # path = Path(f"data/data/{args.path}/01_ERR_SEG_TRA")
    # video = sorted(path.iterdir())
    # out_path = path.parent / "SEG_HEATMAP"
    # out_path.mkdir(exist_ok=True)
    # idx = 0
    # for img_path in tqdm(video, desc="Creating heatmaps..."):
    #     idx += 1
    #     img = tiff.imread(img_path)
    #     hm = mask_to_gaussian(img)
    #     tiff.imwrite(f"{str(out_path)}/heatmap_{idx:02d}.tif", hm)
