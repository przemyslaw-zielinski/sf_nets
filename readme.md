# Slow-fast nets

Code for the paper:
"Discovery of slow variables in a class of multiscale stochastic systems
via neural networks."


## Running experiments

All experiment scripts are located in `scripts/` folder.
Run them from the root directory
```
$python scripts/eperiment_file.py
```
If the script produces some output (figure, table, etc.), it is stored in an
appropriate subdirectory of `results/` folder.


## Training the networks

To train the network, first create the `yaml` config file and store it in `configs/`
folder. Then use the `train.py` script passing the path to the config
script
```
$python train.py configs/my_config.yaml
```


## Config files

Configs to train on a given `dataset` are stored in `configs/dataset` subfolder.
You need to specify which classes will be used for

* `dataset`
* `model`
* `optimizer`
* `trainer`

Every field takes two keys: `type` to select the specific class and `args` to
pass arguments for initialization.


## Project structure

```
sf_nets/
│
├── configs/ - configuration files (.yaml) for training
│
├── data/ - default directory for storing input data
│   ├── dataset1
│       ├── processed
│       ├── raw
│
├── notebooks/
│
├── results/
|   ├── figs/ - figures produced by code from scripts/ or notebooks/ folder
│   ├── models/ - where trained models and their checkpoints go
│   ├── logs/
|   ├── tabs/ - latex tables produced by code from scripts/ or notebooks/ folder
│   ├── reports/ - for storing analyses
│
├── scripts/ - source files for experiments
|
├── sf_nets/ - source files for main classes
│   ├── datasets/
│   ├── loaders/
│   ├── models/
│   ├── trainers/
│
├── spaths/ - separate module for simulating stochastic systems
│
├── utils/
│
├── environment.yaml - packages specification to reproduce conda environment
├── experiments.yaml - database of models used by the experiment scripts
├── train_batch.py - script to select multiple models to train at once
├── train.py - main script to start training
```


## Classes

BaseTrainer:
1. Stores `model`, `loss` and `optimizer` as attributes.
2. Constructs `train_loader` and (optionally) `valid_loader` from `dataset`.
3. Requires each subclass to implement `_train_epoch` method for training logic.
4. Loops over epochs and updates the best model.
5. Saves checkpoints and final summary.
6. Prints and stores logs.


## Acknowledgements
This project structure is inspired by the project [PyTorch Template Project](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque).
