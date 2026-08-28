# Deep flux optique

📦 Installation
------------------




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

