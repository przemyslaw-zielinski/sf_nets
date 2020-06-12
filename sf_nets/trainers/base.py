#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 8 Jun 2020

@author: Przemyslaw Zielinski
"""

import os
import torch
from copy import deepcopy
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, random_split

class BaseTrainer(ABC):

    def __init__(self, dataset, model, cost, optimizer, **config):

        self.dataset = dataset
        self.model = model
        self.cost = cost
        self.optimizer = optimizer
        self.config = config

        self.init_loaders(DataLoader)

        self.history = {
            'train_losses': [],
            'valid_losses': []
        }

        self.state = { # best fit
            'args': model.args_dict,
            'features': model.features,
            'best_epoch': 1,
            'best_model_dict': {},
            'best_optim_dict': {},
            'last_model_dict': {},
            'last_optim_dict': {}
        }

    def __init_subclass__(cls):
        # checks if a subclass implements validation logic
        if "_valid_epoch" in cls.__dict__:
            cls._use_valid = True
        else:
            cls._use_valid = False
            print(f"No validation logic for {cls.__name__}!")

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        Arguments:
            epoch (int): current epoch number

        Returns:
            log (dict): with keys 'training_loss', 'valid_loss',
        """

    def train(self, model_id):

        for epoch in range(1, self.config['max_epochs']+1):

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
            if epoch == 1 or epoch % 10 == 0:
                print(f"epoch : {epoch:3d}/{self.config['max_epochs']}, "
                      f"reconstruction loss = {train_loss:.5f}, "
                      f"validation loss = {valid_loss:.5f}")
        self.state['last_model_dict'] = self.model.state_dict()
        self.state['last_optim_dict'] = self.optimizer.state_dict()
        self._save(model_id)

    def init_loaders(self, Loader):
        # TODO: make more general and check if vaalidation logic exists
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
        # update best fit based on validation performance
        curr_acc = self.history['valid_losses'][-1]
        best_acc = self.history['valid_losses'][self.state['best_epoch']-1]
        if epoch > 1 and curr_acc < best_acc:
            state = {
                'best_epoch': epoch,
                'best_model_dict': deepcopy(self.model.state_dict()),
                'best_optim_dict': deepcopy(self.optimizer.state_dict())
            }
            self.state.update(state)

    def _save(self, model_id):
        system = self.dataset.name
        # model_id = self.config['model_id']
        path = f'results/models/{system}'
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({'config': self.config,
                    'history': self.history,
                    'state': self.state},
                    f'{path}/{model_id}.pt')
