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

The raw datasets does not contains segmentation with tracking labels. But to evaluate the optical flow we will need those, use the following command to create a ERR_SEG_TRA folder that contains segmentation (from SEG_TRA) with temporal consistent label (from GT/TRA).
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


## Reproduce 

### Training 

To train a model use : 

``` bash
python scripts/train_cells.py
```

For example : 

``` bash
python scripts/train_cells.py --mlflow --model_type flowseg
```

### Testing

To test a model use : 

``` bash
python scripts/tests.py
```

For example : 

``` bash
python scripts/tests.py --mlflow --visualize --model_type flowseg
```

The model that will be tested will be the best.pt in the output directory given. The model type given must correspond to the output directory given.
When testing pretrain model or analytic method, the output dir will be outputs/{name of the model} (anything can be put as an argument for output_dir).
To test raft you must have a folder named raft_models containing the weights.

## Methodology

