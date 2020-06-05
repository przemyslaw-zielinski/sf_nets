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

def train(config):

    # TODO: init data loader (which inits dataset ?)
    dataset = init_object(config["dataset"], module_datasets)
    print(f"{dataset.name = }")

    arch = {
        'input_features': dataset.ndim,
        'latent_features': dataset.sdim
        }
    model = init_object(config["model"], module_models, **arch)
    print(f"{model = }")

    loss_func = init_object(config["loss_func"], module_models)
    print(f"{loss_func = }")

    optimizer = init_object(config['optimizer'], torch.optim, model.parameters())
    print(f"{optimizer = }")

    trainer = init_object(config['trainer'], module_trainers)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def init_object(config, module, *args, **kwargs):

    module_type = config["type"]
    module_args = config["args"]
    module_args.update(kwargs)

    return getattr(module, module_type)(*args, **module_args)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('config', type=str,
                      help='config file path')
    args = args.parse_args()

    config = read_json(args.config)
    train(config)
