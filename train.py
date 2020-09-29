#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 5 Jun 2020

@author: Przemyslaw Zielinski
"""

import json
import torch
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
import sf_nets.datasets as module_datasets
import sf_nets.trainers as module_trainers
import sf_nets.models as module_models

def train(model_id, config):

    torch.manual_seed(hash_to_int(model_id))
    torch.randint(10**5, (10**3,))  # warm-up of rng

    logger = logging.getLogger('sf_nets')
    logger.setLevel(logging.DEBUG)

    # configure console handler
    c_handler = logging.StreamHandler()
    c_format = logging.Formatter('') # '%(name)s : %(message)s')
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
                f'\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                '\n####################\n')

    # TODO: init data loader (which inits dataset ?)
    dataset = init_object(config['dataset'], module_datasets)
    logger.info(f'Loaded dataset:\t{dataset}\n')

    default_inla = { # takes correct dims from dataset metadata
        'input_features': dataset.system.ndim,
        'latent_features': dataset.system.sdim
        } # bu can be overrided in config
    config['model']['args'] = {**default_inla, **config['model']['args']}
    model = init_object(config['model'], module_models)
    logger.info(f'Loaded model:\t{model}\n')
    # logger.debug(list(model.parameters())[0])

    # loss_func = init_object(config['loss_function'], module_models)
    # logger.info(f'Loaded loss:\t{loss_func}\n')

    optimizer = init_object(config['optimizer'], torch.optim, model.parameters())
    logger.info(f'Loaded optimizer:\t{optimizer}\n')

    scheduler = config.get('scheduler', None)
    if scheduler is not None:
        scheduler = init_object(scheduler, torch.optim.lr_scheduler, optimizer)
        logger.info(f'Loaded scheduler:\t{scheduler}\n')

    trainer = init_object(config['trainer'], module_trainers,
                          dataset, model, optimizer, scheduler)
    logger.info(f'Loaded trainer:\t{trainer}\n')

    logger.info("TRAINING LOOP")
    trainer.train(model_id)

def read_json(fpath):
    fpath = Path(fpath)
    with fpath.open('rt') as json_file:
        return fpath.stem, json.load(json_file, object_hook=dict)

def hash_to_int(string):
    bstring = string.encode()
    return int(hashlib.sha1(bstring).hexdigest(), 16) % 10**16

def init_object(config, module, *args, **kwargs):

    module_type = config["type"]
    module_kwargs = config["args"]
    module_kwargs.update(kwargs)

    return getattr(module, module_type)(*args, **module_kwargs)

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('config', type=str,
                      help='Config file path')
    args.add_argument('-r', '--replica', type=int, default=0,
                      help='New realization id (>=1)')
    args = args.parse_args()

    model_id, config = read_json(args.config)
    if args.replica > 0:
        model_id += f'_r{args.replica}'

    train(model_id, config)
