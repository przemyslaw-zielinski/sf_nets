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

    # log_path = Path('results/logs') / config['dataset']['type']
    logger = init_logger(
        'sf_nets',
        f"results/logs/{config['dataset']['type']}",
        model_id
    )

    logger.info(
        '####################'
        f'\nTRAINING {model_id}'
        f'\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        '\n####################\n'
    )

    # TODO: init data loader (which inits dataset ?)
    dataset = init_object(config['dataset'], module_datasets)
    logger.info(f'Loaded dataset:\t{dataset}\n')

    default_inla = { # takes correct dims from dataset metadata
        'input_features': dataset.ndim,
        'latent_features': dataset.sdim
        } # bu can be overrided in config
    config['network']['args'] = {**default_inla, **config['network']['args']}
    network = init_object(config['network'], module_models)
    logger.info(f'Loaded network:\t{network}\n')
    # logger.debug(list(network.parameters())[0])

    loss_func = init_object(config['loss_function'], module_models)
    logger.info(f'Loaded loss:\t{loss_func}\n')

    optimizer = init_object(config['optimizer'], torch.optim, network.parameters())
    logger.info(f'Loaded optimizer:\t{optimizer}\n')

    scheduler = config.get('scheduler', None)
    if scheduler is not None:
        scheduler = init_object(scheduler, torch.optim.lr_scheduler, optimizer)
        logger.info(f'Loaded scheduler:\t{scheduler}\n')

    trainer = init_object(config['trainer'], module_trainers,
                          dataset, network, loss_func, optimizer, scheduler)
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

def init_logger(name, log_path, file_name):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # TODO: this can create dir for nonexistent dataset
    log_path = Path(log_path)
    log_path.mkdir(exist_ok=True)
    log_file = log_path / f'{file_name}.log'

    # configure console handler
    c_handler = logging.StreamHandler()
    c_format = logging.Formatter('') # '%(name)s : %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # configure file handler
    f_handler = logging.FileHandler(log_file, mode='w')
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    return logger

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
