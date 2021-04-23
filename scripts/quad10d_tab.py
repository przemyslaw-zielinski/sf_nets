#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 19 Mar 2021

@author: Przemyslaw Zielinski
"""

import sys, os
sys.path[0] = os.getcwd()

import yaml
import torch
from itertools import chain
from tabulate import tabulate
import sf_nets.models as models
from utils.io_utils import io_path, get_script_name

ds_name = 'Quad10'
script_name = get_script_name()
io_path = io_path(ds_name)

def get_sparsity_per_layer(encdec):
    sparsities = []
    for module in chain(encdec.encoder, encdec.decoder):
        num = 0.0
        den = 0.0
        if hasattr(module, 'weight'):
            num += torch.sum(module.weight == 0)
            den += module.weight.nelement()
        if hasattr(module, 'bias') and module.bias is not None:
            num += torch.sum(module.bias == 0)
            den += module.bias.nelement()

        if den > 0:
            sparsities.append(float(num) / float(den))
    return sparsities

with open("experiments.yaml", 'rt') as yaml_file:
    models_data = yaml.full_load(yaml_file)[ds_name]
model_ids = models_data['model_ids']
model_tags = models_data['model_tags']

table = []
headers = ["Model", "Layer sizes", "Max. epochs", "Min. validation loss",
    # "Sparsity per layer [%]",
     # "Total sparsity [%]"
     ]

for id, tag in zip(model_ids, model_tags):

    model_data = torch.load(io_path.models / f'{id}.pt')
    # model_arch = model_data['info']['architecture']
    # model_args = model_data['info']['arguments']
    # state_dict = model_data['best']['model_dict']
    #
    # model = getattr(models, model_arch)(**model_args)
    # model.load_state_dict(state_dict)
    # model.eval()

    features = model_data['info']['features']
    features = " - ".join([str(f) for f in features])
    max_epochs = model_data['info']['config']['max_epochs']
    best_epoch = model_data['best']['epoch']
    min_val_loss = model_data['history']['train_losses'][best_epoch-1]
    # sparsities = get_sparsity_per_layer(model)
    # features = " - ".join(f"{f}({int(100*sp)})" for f, sp in zip(features, sparsities))
    # sparsities = " - ".join([str(int(100*sp)) for sp in sparsities])

    # tot_sparsity = int(100* model.sparsity)
    table.append([tag, features, max_epochs, min_val_loss])

file_path = io_path.tabs / f"{ds_name}_tab.tex"
with file_path.open('w') as file:
    file.write(
        tabulate(
            table,
            headers=headers,
            tablefmt='latex_booktabs',
            floatfmt=".4f",
            colalign=("left","center",)
        )
    )

table = []
headers = ["Model",  "Sparsity per layer [%]", "Total sparsity [%]"]

for id, tag in zip(model_ids, model_tags):

    if "pruned" in id:
        model_data = torch.load(io_path.models / f'{id}.pt')
        model_arch = model_data['info']['architecture']
        model_args = model_data['info']['arguments']
        state_dict = model_data['best']['model_dict']

        model = getattr(models, model_arch)(**model_args)
        model.load_state_dict(state_dict)
        model.eval()

        # features = model_data['info']['features']
        # features = " - ".join([str(f) for f in features])
        # max_epochs = model_data['info']['config']['max_epochs']
        # best_epoch = model_data['best']['epoch']
        # min_val_loss = model_data['history']['train_losses'][best_epoch-1]
        sparsities = get_sparsity_per_layer(model)
        features = " - ".join(f"{f}({int(100*sp)})" for f, sp in zip(features, sparsities))
        sparsities = " - ".join([str(int(100*sp)) for sp in sparsities])

        tot_sparsity = int(100* model.sparsity)
        table.append([tag, sparsities, tot_sparsity])

file_path = io_path.tabs / f"{ds_name}_sparsitytab.tex"
with file_path.open('w') as file:
    file.write(
        tabulate(
            table,
            headers=headers,
            tablefmt='latex_booktabs',
            floatfmt=".4f",
            colalign=("left","center",)
        )
    )
