#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 8 Jun 2020

@author: Przemyslaw Zielinski
"""

import os
import torch
import shutil
import logging
from pathlib import Path
from copy import deepcopy
from abc import ABC, abstractmethod
from utils import tb_utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

class BaseTrainer(ABC):

    can_validate = False

    def __init__(self, dataset, model, optimizer, scheduler=None, **config):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        self.max_epochs = config['max_epochs']
        self.checkpoints = set(range(config['checkpoint_start'],
                                     config['max_epochs'] + 1,
                                     config['checkpoint_freq']))

        self.model.set_system(dataset.system)  # TODO: improve this bit
        self.init_loaders(config.get('loader', DataLoader))  # TODO: make it work!

        self.history = {
            'train_losses': [],
            'checkpoints': []  # (epoch, type) where type = {state, best}
        }
        if self.can_validate:
            self.history['valid_losses'] = []

        # dataset directory for storing checkpoints
        # self.path = Path(f'results/models/{self.dataset.name}')
        self.path.mkdir(exist_ok=True)

        self.best = {
            'epoch': 1,
            'model_dict': {},
            'optim_dict': {},
        }

    def __repr__(self):
        return type(self).__name__

    def __init_subclass__(cls):
        # checks if a subclass implements validation logic
        if "_valid_epoch" in cls.__dict__:
            cls.can_validate = True
        else:
            print(f"No validation logic for {cls.__name__}!")

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

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        Arguments:
            epoch (int): current epoch number

        Returns:
            train_loss (float): training loss after the current epoch
        """

    def train(self, model_id):

        # TODO: use ckpt(s) for checkpoint(s)
        self.cpdir = self.path / f'{model_id}'
        makedir(self.cpdir)  # TODO: use property?
        writer = SummaryWriter(self.cpdir, max_queue=100)

        for epoch in range(1, self.max_epochs + 1):

            width = len(str(self.max_epochs))
            log_msg = [f'epoch : {epoch:{width}d}/{self.max_epochs}']

            train_loss = self._train_epoch(epoch)
            self.history['train_losses'].append(train_loss)
            writer.add_scalar('Loss/train', train_loss, epoch)
            log_msg.append(f'reconstruction loss = {train_loss:.5f}')
            # TODO: how to pass additional information?
            # train_loss, *train_log = self._train_epoch(epoch)
            # valid_loss, *valid_log = self._valid_epoch(epoch)
            if self.can_validate:
                # compute the epoch validation loss
                with torch.no_grad():
                    valid_loss = self._valid_epoch(epoch)
                    self.history['valid_losses'].append(valid_loss)
                    writer.add_scalar('Loss/validation', valid_loss, epoch)
                    log_msg.append(f'validation loss = {valid_loss:.5f}')

            if self.scheduler is not None:
                self.scheduler.step()
                if epoch % self.scheduler.step_size == 0:
                    lrs = [pg['lr'] for pg in self.optimizer.param_groups]
                    log_msg.append(f'learning rate(s) = {lrs}')

            if epoch in self.checkpoints or epoch == self.max_epochs:
                self.logger.info(', '.join(log_msg))
                self._save_checkpoint(epoch)  # TODO: add additional info

                for log_name, kwargs in self.config['tb_logs']:
                    log_fn = getattr(tb_utils, log_name)
                    log_fn(writer, self, epoch, **kwargs)
                    
            self._update_best(epoch)

        self._save_checkpoint(epoch, best=True)
        self._save(model_id)
        writer.close()

    def init_loaders(self, Loader):
        # TODO: make more general
        valid_size = int(self.config.get('valid_split', 0) * len(self.dataset))
        if valid_size == 0 and self.can_validate == True:
            raise ValueError('Specify positive valid_split parameter'
                             'in trainer args!')
        train_size = len(self.dataset) - valid_size

        if valid_size > 0 and self.can_validate == False:
            self.logger.warning('Validation logic not available! Skipping this part.')
            valid_size = 0
            train_size = len(self.dataset)

        train_data, valid_data = random_split(self.dataset, [train_size, valid_size])
        self.train_loader = Loader(train_data,
                                       batch_size=self.config['batch_size'],
                                       shuffle=self.config['shuffle'])
        self.logger.info('Initialized training loader with '
                         f'{train_size} samples.')
        if len(valid_data) > 0:
            self.valid_loader = Loader(valid_data,
                                       batch_size=self.config['batch_size'],
                                       shuffle=self.config['shuffle'])
            self.logger.info('Initialized validation loader with '
                             f'{valid_size} samples.')

        loader_config = {
            'type': Loader.__class__.__name__,
            'args': {},
            'train_size': train_size,
            'valid_size': valid_size
        }
        self.config['loader'] = loader_config

    def _update_best(self, epoch):

        if valid_losses := self.history.get('valid_losses'):
            # update best fit based on validation performance
            curr_acc = valid_losses[-1]
            best_acc = valid_losses[self.best['epoch']-1]
            if epoch == 1 or curr_acc >= best_acc:
                return
        elif epoch < self.max_epochs + 1:
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
                        'best': self.best,  # TODO: remove
                        'history': self.history
                    },
                    self.path / f'{model_id}.pt')

class SimpleTrainer(BaseTrainer, ABC):

    def __init_subclass__(cls):
        super.__init_subclass__()

    def __repr__(self):
        return type(self).__name__

    @abstractmethod
    def _compute_loss(self, x, x_model, *x_dat):
        pass

    def _train_epoch(self, epoch):

        epoch_loss = 0.0
        for x, *x_dat in self.train_loader:

            self.optimizer.zero_grad()
            batch_loss = self._compute_loss(x, self.model(x), *x_dat)
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()

        return epoch_loss / len(self.train_loader)

    def _valid_epoch(self, epoch):

        epoch_loss = 0.0
        for x, *x_dat in self.valid_loader:
            epoch_loss += self._compute_loss(x, self.model(x), *x_dat).item()

        return epoch_loss / len(self.valid_loader)

def makedir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
