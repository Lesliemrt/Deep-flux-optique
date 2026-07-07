# Deep flux optique

## Dataset description

The datasets used in this project are from the Cell Tracking Challenge web site (https://celltrackingchallenge.net/) and are all 2D datasets.
Each dataset folder contains 2 videos (01 and 02), one is used for training and the second for testing. 
Here is a description of the different type of annotation used in this project (with N = 1 or 2) :
 - 0N_ERR_SEG : imperfect segmentation masks for sequence of the real dataset (labels without temporal coherence)
 - 0N_GT : gold truth for the sequence
    - SEG : gold segmentation truth, not complete
    - TRA : gold tracking truth

The different dataset used are : 
 - Fluo-N2DH-GOWT1_train : GFP-GOWT1 mouse stem cells, Dr. E. Bártová. Institute of Biophysics, Academy of Sciences of the Czech Republic, Brno, Czech Republic
 - BF-C2DL-MuSC : Mouse muscle stem cells in hydrogel microwells, Dr. H. Blau, Baxter Laboratory for Stem Cell Biology, Stanford University, USA
 - Fluo-N2DL-HeLa_train : HeLa cells on a flat glass, Dr. G. van Cappellen. Erasmus Medical Center, Rotterdam, The Netherlands
 - BF-C2DL-HSC : Mouse hematopoietic stem cells in hydrogel microwells, Dr. H. Blau, Baxter Laboratory for Stem Cell Biology, Stanford University, USA

 ## Data preprocessing

The raw datasets does not contains segmentation with tracking labels. But to evaluate the optical flow we will need those, use the following command to create a ERR_SEG_TRA folder that contains segmentation with temporal consistent label.
``` bash
# Use --path to specify the dataset to preprocess, for exemple :
 python scripts/datapreprocessing.py --path "data/Fluo-N2DH-GOWT1"

```

## Project structure

```bash
|-- data/
| |--


```

## Installation

First clone the repository :

```bash 
git clone https://github.com/Lesliemrt/Deep-flux-optique.git

```

Then install the requirements :

```bash 
conda env create -f environment_no_builds.yml
```
or with :

```bash 
pip install -r requirements.txt
```

In the config.py file, you will need to set variables :

 - TRAIN_DATA : video to train the model on
 - TRAIN_SEG_DATA : segmentation ground truth for the training video
 - TEST_DATA : video to test the model on
 - TEST_SEG_DATA : segmentation ground truth for the testing video
 - HEIGHT : height of images
 - WIDTH : width of images

## Reproduce 

### Training 

To train a model use : 

``` bash
python scripts/train_cells.py
```
| Argument | Type | Default value | Possible choices | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--mlflow` | `flag` | `False` | N/A | Enable mlflow tracking during training |
| `--workers` | `int` | `4` | *Any integer* | Number of workers |
| `--output-dir` | `str` | `"outputs"` | *Any string* | Output directory |
| `--video-path` | `str` | `TRAIN_DATA` | *Any string* | Path to the training video |
| `--seg-path` | `str` | `TRAIN_SEG_DATA` | *Any string* | Path to the segmentation of the training video, if not available set to None |
| `--max_gap` | `int` | `20` | *Any integer* | Maximum temporal gap between 2 frames |
| `--multiplier` | `int` | `3` | *Any integer* | Multiplier to have more pairs per epoch |
| `--seg_type` | `str` | `"binary"` | `"binary"`, `"heatmaps"`, `None` | Type of segmentation to use for training: None, 'binary' or heatmaps |
| `--model_type` | `str` | `"voxelmorph"` | `"voxelmorph"`, `"twoflows"`, `"flowseg"` | Type of model to train : voxelmorph, twoflows flowseg |
| `--epochs` | `int` | `30` | *Any integer* | Number of epochs |
| `--batch-size` | `int` | `4` | *Any integer* | Batch size |
| `--save-every` | `int` | `10` | *Any integer* | Checkpoint every N epochs |
| `--cycle_param` | `float` | `0.005` | *Any float* | Regularisation parameter for cycle loss |
| `--image-loss` | `str` | `"mse"` | `"mse"`, `"cc"` | Image loss to use: 'mse' or 'cc' |
| `--grad_loss` | `str` | `"l2"` | `"l1"`, `"l2"` | Gradient loss to use: 'l1' or 'l2' |
| `--lr` | `float` | `1e-3` | *Any float* | Learning rate |
| `--lambda` | `float` | `0.1` | *Any float* | Regularisation parameter for smooth loss |
| `--gamma` | `float` | `0.01` | *Any float* | Regularisation parameter for dice loss between source and warped target, not used if seg_path is set to None |
| `--seg_param` | `float` | `0.01` | *Any float* | Regularisation parameter for segmentation loss, not used if seg_path is set to None |
| `--background` | `flag` | `False` | N/A | Add loss to avoid displacement in the background, only available if segmentations available |
| `--background_param`| `float` | `0.01` | *Any float* | Parameter for background loss, only available if segmentations available |

For example : 

``` bash
python scripts/train_cells.py --mlflow --model_type twoflows
```

### Testing

To train a model use : 

``` bash
python scripts/tests.py
```

| Argument | Type | Default value | Possible choices | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--mlflow` | `flag` | `False` | N/A | Enable mlflow tracking during training |
| `--visualize` | `flag` | `False` | N/A | Enable visualization of flow maps |
| `--video_path` | `str` | `TEST_DATA` | *Any string* | Moving image (source) filename |
| `--gap` | `int` | `1` | *Any integer* | Temporal gap between 2 frames |
| `--seg_path` | `str` | `TEST_SEG_DATA` | *Any string* | Segmentations for evaluation (if segmentations not available, set to None) |
| `--model_type` | `str` | `"voxelmorph"` | `"voxelmorph"`, `"twoflows"`, `"flowseg"`, `"tvl1"`, `"raft"`, `"farnback"` | Type of model to use for registration |
| `--output_dir` | `str` | `"outputs/flowseg/output_20260707_152433"` | *Any string* | Directory to find model and to save results |

For example : 

``` bash
python scripts/tests.py --mlflow --visualize --model_type flowseg
```

The model that will be tested will be the best.pt in the output directory given. The model type given must correspond to the output directory given.
When testing pretrain model or analytic method, the ouotput dir will be outputs/{name of the model} and anything can be put as an argument for output_dir.
To test raft you must have a folder named raft_models