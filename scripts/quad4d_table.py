#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 18 Mar 2021

@author: Przemyslaw Zielinski
"""

import sys, os
sys.path[0] = os.getcwd()

import yaml
import torch
from tabulate import tabulate
from utils.io_utils import io_path, get_script_name

ds_name = 'Quad4'
script_name = get_script_name()
io_path = io_path(ds_name)

with open("experiments.yaml", 'rt') as yaml_file:
    models_data = yaml.full_load(yaml_file)[ds_name]
model_ids = models_data['model_ids']
model_tags = models_data['model_tags']

table = []
headers = ["Model", "Layer sizes", "Max. epochs", "Min. validation loss"]
for id, tag in zip(model_ids, model_tags):

    model_data = torch.load(io_path.models / f'{id}.pt')

    features = " - ".join([str(f) for f in model_data['info']['features']])
    max_epochs = model_data['info']['config']['max_epochs']
    best_epoch = model_data['best']['epoch']
    min_val_loss = model_data['history']['train_losses'][best_epoch]
    table.append([tag, features, max_epochs, min_val_loss])

file_path = io_path.tabs / f"{ds_name}_tab.tex"
with file_path.open('w') as file:
    file.write(
        tabulate(
            table,
            headers=["Model", "Layer sizes", "Max. epochs", "Min. validation loss"],
            tablefmt='latex_booktabs',
            floatfmt=".4f",
            colalign=("left","center",)
        )
    )
