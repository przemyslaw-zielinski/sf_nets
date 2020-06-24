Code for the discovery of slow variables in high-dimensional stochastic systems
via autoencoders.


Project structure

sf_nets/
|
|-- configs/ - directory for storing configuration files (.json) for training
|
|-- data/ - default directory for storing input data
|
|-- models/
|
|-- notebooks/
|
|-- results/
|   |-- models/
|   |-- logs/
|   |-- reports/ - for storing analyses
|
|-- sf_nets/
|   |-- datasets/
|   |-- loaders/
|   |-- models/
|   |-- trainers/
|
|-- utils/
|
|-- train.py - main script to start training

BaseTrainer:
1. Loops over epochs.
2. Updates the best model.
3. Prints and stores logs.
4. Saves checkpoints and final summary.

TODO
[X] add checkpoints
[ ] checkpoints frequency
[ ] add stopping criterions
[ ] add logger
[ ] overwrite or make replica of model ({model_id}r.pt)
