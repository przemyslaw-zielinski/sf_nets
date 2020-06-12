#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 5 Jun 2020

@author: Przemyslaw Zielinski
"""

import torch
import json
import argparse
from pathlib import Path
from collections import OrderedDict
import sf_nets.datasets as module_datasets
import sf_nets.trainers as module_trainers
import sf_nets.models as module_models

def train(model_id, config):

    # TODO: init data loader (which inits dataset ?)
    dataset = init_object(config['dataset'], module_datasets)
    print(f"{dataset.name = }")

    arch = {
        'input_features': dataset.ndim,
        'latent_features': dataset.sdim
        }
    model = init_object(config['model'], module_models, **arch)
    print(f"{model = }")

    loss_func = init_object(config['loss_func'], module_models)
    print(f"{loss_func = }")

    optimizer = init_object(config['optimizer'], torch.optim, model.parameters())
    print(f"{optimizer = }")

    trainer = init_object(config['trainer'], module_trainers,
                          dataset, model, loss_func, optimizer)

    print("Training loop.")
    trainer.train(model_id)

def read_json(fpath):
    fpath = Path(fpath)
    with fpath.open('rt') as json_file:
        return fpath.stem, json.load(json_file, object_hook=dict)

def init_object(config, module, *args, **kwargs):

    module_type = config["type"]
    module_kwargs = config["args"]
    module_kwargs.update(kwargs)

    return getattr(module, module_type)(*args, **module_kwargs)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('config', type=str,
                      help='config file path')
    args = args.parse_args()

    model_id, config = read_json(args.config)
    train(model_id, config)
