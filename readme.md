# Slow-fast autoencoders

Code for the discovery of slow variables in high-dimensional stochastic systems
via autoencoders.


## Project structure

```
sf_nets/
│
├── configs/ - directory for storing configuration files (.json) for training
│
├── data/ - default directory for storing input data
│
├── notebooks/
│
├── results/
│   ├── models/ - where trained models and their checkpoints go
│   ├── logs/
│   ├── reports/ - for storing analyses
│
├── sf_nets/ - source files
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
1. Loops over epochs.
2. Updates the best model.
3. Prints and stores logs.
4. Saves checkpoints and final summary.

## TODOs
- [X] add checkpoints
- [ ] checkpoints frequency
- [ ] add stopping criterions
- [ ] add logger
- [ ] seed setting
- [ ] overwrite or make replica of model ({model_id}r.pt)

## Acknowledgements
This project structure is inspired by the project [PyTorch Template Project](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque).
