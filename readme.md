# Slow-fast autoencoders

Code for the discovery of slow variables in high-dimensional stochastic systems
via autoencoders.


## Project structure

```
sf_nets/
│
├── configs/ - configuration files (.json) for training
│
├── data/ - default directory for storing input data
│   ├── dataset1
│       ├── processed
│       ├── raw
│
├── notebooks/
│
├── results/
│   ├── models/ - where trained models and their checkpoints go
│   ├── logs/
│   ├── reports/ - for storing analyses
│
├── sf_nets/ - source files for main classes
│   ├── datasets/
│   ├── loaders/
│   ├── models/
│   ├── trainers/
│
├── utils/
│
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

## TODOs
- [X] add checkpoints
- [ ] checkpoints frequency
- [ ] add stopping criterions
- [ ] add logger
- [ ] seed setting
- [ ] overwrite or make replica of model ({model_id}r.pt)

## Acknowledgements
This project structure is inspired by the project [PyTorch Template Project](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque).
