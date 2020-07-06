#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 5 Jun 2020

@author: Przemyslaw Zielinski
"""

import json
import torch
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
import sf_nets.datasets as module_datasets
import sf_nets.trainers as module_trainers
import sf_nets.models as module_models

def train(model_id, config):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # configure console handler
    c_handler = logging.StreamHandler()
    c_format = logging.Formatter('')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # TODO: this can create dir for nonexistent dataset
    log_path = Path('results/logs') / config['dataset']['type']
    log_path.mkdir(exist_ok=True)

    # configure file handler
    f_handler = logging.FileHandler(log_path / f'{model_id}.log', mode='w')
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)


    logger.info('####################'
                f'\nTRAINING {model_id}'
                f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                '\n####################\n')

    # TODO: init data loader (which inits dataset ?)
    dataset = init_object(config['dataset'], module_datasets)
    logger.info(f'Loaded dataset:\t{dataset}\n')

    arch = {
        'input_features': dataset.ndim,
        'latent_features': dataset.sdim
        }
    model = init_object(config['model'], module_models, **arch)
    logger.info(f'Loaded model:\t{model}\n')

    loss_func = init_object(config['loss_func'], module_models)
    logger.info(f'Loaded loss:\t{loss_func}\n')

    optimizer = init_object(config['optimizer'], torch.optim, model.parameters())
    logger.info(f'Loaded optimizer:\t{optimizer}\n')

    trainer = init_object(config['trainer'], module_trainers,
                          dataset, model, loss_func, optimizer)
    logger.info(f'Loaded trainer:\t{trainer}\n')

    logger.info("TRAINING LOOP")
    # TODO: add epoch logging
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
