# Deep flux optique

## 📦 Installation

### Install from repo

```bash 
git clone https://github.com/Lesliemrt/Deep-flux-optique.git
cd Deep-flux-optique
pip install .

```

Or install from Pypi :

```bash 

```

* * *


## 👩🏻‍💻 Reproduce 

### Data

The pipeline was originaly created for Cell Tracking Challenge's data.
The names of the files or folders can be different but it must follow this structure : 

```
root_data_folder/
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

### Training 

To train a model use : 

``` bash
python scripts/train_cells.py --video-path root_data_folder/data1/train_data --seg-path root_data_folder/data1/seg_train_data
```

You can specify the model that you want to train using the --model_type argument, for example : 

``` bash
python scripts/train_cells.py --mlflow --model_type flowseg
```

### Inference

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



