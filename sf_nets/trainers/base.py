#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 8 Jun 2020

@author: Przemyslaw Zielinski
"""

import os
import torch
import shutil
from h5py import File
from pathlib import Path
from copy import deepcopy
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, random_split

class BaseTrainer(ABC):

    _use_valid = False

    def __init__(self, dataset, model, cost, optimizer, **config):

        self.dataset = dataset
        self.model = model
        self.cost = cost
        self.optimizer = optimizer
        self.config = config

        self.init_loaders(DataLoader)

        self.history = {
            'train_losses': [],
            'checkpoints': []  # (epoch, type) where type = {state, best}
        }
        if self._use_valid:
            self.history['valid_losses'] = []

        # dataset directory for storing checkpoints
        # self.path = Path(f'results/models/{self.dataset.name}')
        self.path.mkdir(exist_ok=True)

        self.best = {
            'epoch': 1,
            'model_dict': {},
            'optim_dict': {},
        }

    @property
    def info(self):
        return {
            'architecture': type(self.model).__name__,
            'arguments': self.model.args_dict,
            'features': self.model.features,
            'config' : self.config
        }

    @property
    def path(self):
        return Path(f'results/models/{self.dataset.name}')

    def __init_subclass__(cls):
        # checks if a subclass implements validation logic
        if "_valid_epoch" in cls.__dict__:
            cls._use_valid = True
        else:
            print(f"No validation logic for {cls.__name__}!")

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        Arguments:
            epoch (int): current epoch number

        Returns:
            train_loss (float): training loss after 'the current epoch
        """

    def train(self, model_id):

        self.cpdir = self.path / f'{model_id}'
        makedir(self.cpdir)  # TODO: use property?

        for epoch in range(1, self.config['max_epochs']+1):

            log = {'epoch': epoch}

            train_loss = self._train_epoch(epoch)
            self.history['train_losses'].append(train_loss)
            # TODO: how to pass additional information?
            # train_loss, *train_log = self._train_epoch(epoch)
            # valid_loss, *valid_log = self._valid_epoch(epoch)
            if self._use_valid:
                # compute the epoch validation loss
                with torch.no_grad():
                    valid_loss = self._valid_epoch(epoch)
                    self.history['valid_losses'].append(valid_loss)

            self._update_best(epoch)

            # display the epoch loss
            # TODO: implement logger
            if epoch == 1 or epoch % 10 == 0:
                print(f"epoch : {epoch:3d}/{self.config['max_epochs']}, "
                      f"reconstruction loss = {train_loss:.5f}, "
                      f"validation loss = {valid_loss:.5f}")

        self._save_checkpoint(epoch, best=True)
        self._save(model_id)

    def init_loaders(self, Loader):
        # TODO: make more general
        valid_size = int(self.config['valid_split'] * len(self.dataset))
        train_size = len(self.dataset) - valid_size
        loader_config = {
            'type': Loader.__class__.__name__,
            'args': {},
            'train_size': train_size,
            'valid_size': valid_size
        }
        self.config['loader'] = loader_config
        train_data, valid_data = random_split(self.dataset, [train_size, valid_size])
        self.train_loader = Loader(train_data,
                                       batch_size=self.config['batch_size'],
                                       shuffle=self.config['shuffle'])
        self.valid_loader = Loader(valid_data,
                                       batch_size=self.config['batch_size'],
                                       shuffle=self.config['shuffle'])

    def _update_best(self, epoch):

        if valid_losses := self.history.get('valid_losses'):
            # update best fit based on validation performance
            curr_acc = valid_losses[-1]
            best_acc = valid_losses[self.best['epoch']-1]
            if epoch == 1 or curr_acc >= best_acc:
                return
        elif epoch < self.config['max_epochs'] + 1:
            # if no validation logic, update last model
            return

        self.best.update({
            'epoch': epoch,
            'model_dict': deepcopy(self.model.state_dict()),
            'optim_dict': deepcopy(self.optimizer.state_dict())
        })

    def _save_checkpoint(self, epoch, info={}, best=False):

        if best:
            checkpt = self.best
            id = f'best_of_{epoch}.pt'
            self.history['checkpoints'].append((checkpt['epoch'], 'best'))
        else:
            checkpt = {
                'epoch': epoch,
                'model_dict': self.model.state_dict(),
                'optim_dict': self.optimizer.state_dict()
            }
            id = f'state_at_{epoch}.pt'
            self.history['checkpoints'].append((epoch, 'state'))

        checkpt.update(info)
        torch.save(checkpt, self.cpdir / id)

    def _save(self, model_id):
        torch.save({
                        'id': model_id,
                        'info': self.info,
                        'best': self.best,
                        'history': self.history
                    },
                    self.path / f'{model_id}.pt')

def makedir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
