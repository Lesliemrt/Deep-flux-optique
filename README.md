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

``` bash
# To create label consistent in time from CTC data
# This will create a folder data/data1/02_ERR_SEG_TRA by combining the masks from data/data1/02_ERR_SEG and the labels from data/data1/02_GT/TRA.
python scripts/datapreprocessing.py --relabel --dataset data1

# To create segmentations with cellpose for training
# This will create a folder data/data1/CELLPOSE_SEG by segmenting data/data1/01.
python scripts/datapreprocessing.py --cellpose --dataset data1
```

To use these commands, your data must follow this structure : 

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

### Training 

To train a model use : 

``` bash
python scripts/train_cells.py --video-path root_data_folder/data1/train_data --seg-path root_data_folder/data1/seg_train_data
```

You can specify the model that you want to train using the --model_type argument, for example : 

``` bash
python scripts/train_cells.py --model_type flowseg
```

### Inference

To test a model use the command below, *output_dir* being the folder created when training a model.

``` bash
python scripts/tests.py --video-path root_data_folder/data1/test_data --seg-path root_data_folder/data1/seg_test_data --output_dir outputs/transflowseg/output_20260728_084107 --model_type transflowseg
```



