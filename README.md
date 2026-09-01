# Deep flux optique

## 📦 Installation

### Install from repo

```bash 
git clone https://github.com/Lesliemrt/Deep-flux-optique.git
cd Deep-flux-optique
pip install .

```

* * *


## 👩🏻‍💻 Reproduce 

### Data

The pipeline was originaly created for Cell Tracking Challenge's data.
The names of the files or folders can be different but it must follow this structure : 

```
data/
├── data1/
│   ├── train_data/
│   │   ├── t000.tiff
│   │   ├── t001.tiff
│   │   └── ...
│   ├── seg_train_data/
│   │   ├── mask000.tiff
│   │   ├── mask001.tiff
│   │   └── ...
│   ├── test_data/
│   │   ├── t000.tiff
│   │   ├── t001.tiff
│   │   └── ...
│   ├── seg_test_data/
│   │   ├── mask000.tiff
│   │   ├── mask001.tiff
│   │   └── ...
├── data2/
│   ├── train_data/
│   ├── seg_train_data/
│   ├── test_data/
│   └── seg_test_data/
├── data3/
...

```

To evaluate the models, the segmentations given must have label consistent in time (tracking labels).

### Data preprocessing

If you use the data from the CTC, you may not have access to segmentation annotations consistent in time. 
You can create it by using :

``` bash
python datapreprocessing.py --dataset data1
```

To use this command, your data must follow this structure : 

```
data/
├── data1/
│   ├── 01/
│   ├── 01_ERR_SEG/
│   ├── 01_GT/TRA
│   ├── 02/
│   ├── 02_ERR_SEG/
│   ├── 02_GT/TRA
├── data2/
...

```
This command will create a folder data/data1/01_ERR_SEG_TRA by combining the masks from data/data1/01_ERR_SEG and the labels from data/data1/01_GT/TRA.

### Training 

To train a model use : 

``` bash
python scripts/train_cells.py --video-path root_data_folder/data1/train_data --seg-path root_data_folder/data1/seg_train_data
```

You can specify the model that you want to train using the --model_type argument, for example : 

``` bash
python scripts/train_cells.py --model_type flowseg
```

The training will create a folder containing the weights of the model. This directory 

### Inference

To test a model use : 

``` bash
python scripts/tests.py --video-path root_data_folder/data1/test_data --seg-path root_data_folder/data1/seg_test_data --output
```

For example : 

``` bash
python scripts/tests.py --model_type flowseg
```

The model that will be tested will be the best.pt in the output directory given. The model type given must correspond to the output directory given.



