# personnal tetst file # noqa: INP001

import matplotlib.pyplot as plt
import tifffile as tiff

img_path = "data/train_data/Fluo-N2DH-GOWT1_train/01/t000.tif"
seg_path = "data/train_data/Fluo-N2DH-GOWT1_train/01_GT/SEG/man_seg012.tif"  # ground truth or gold truth for segmentation annotations  # noqa: E501
tra_path = "data/train_data/Fluo-N2DH-GOWT1_train/01_GT/TRA/man_track000.tif"  # gt for tracking annotations
err_seg_path = "data/train_data/Fluo-N2DH-GOWT1_train/01_ERR_SEG/mask000.tif"  # segmentation to use
st_path = "data/train_data/Fluo-N2DH-GOWT1_train/01_ST/SEG/man_seg000.tif"  # silver truth

img = tiff.imread(img_path)
seg = tiff.imread(seg_path)
tra = tiff.imread(tra_path)
err_seg = tiff.imread(err_seg_path)
st = tiff.imread(st_path)

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.subplot(1, 3, 2)
plt.imshow(tra)
plt.subplot(1, 3, 3)
plt.imshow(err_seg)
plt.show()
