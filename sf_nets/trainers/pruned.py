#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 5 Jun 2020

@author: Przemyslaw Zielinski
"""

import torch
import torch.nn.utils.prune as prune
from utils.dmaps import ln_covs, lnc_ito
from itertools import chain

from copy import deepcopy
from .base import BaseTrainer

class PrunedTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.pruning = self.config['pruning']

        self.prune_func = getattr(prune, self.pruning['type'])
        self.prune_args = self._parse_prune(self.pruning['args'])

        self.start, self.freq = self.pruning['schedule']
        self.sparsity = 0.0

    def _train_epoch(self, epoch):

        train_loss = 0.0
        for x, x_dat in self.train_loader:

            self.optimizer.zero_grad()
            loss = self._loss(x, x_dat, self.model(x))
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        if epoch >= self.start and epoch % self.freq == 0 and self.sparsity < self.pruning['target_sparsity']:
            self._prune()
            print(f'epoch : {epoch}, pruned!, sparsity = {self.sparsity:.2f}')

        if epoch % 15 == 0:
            self._save_checkpoint(epoch, info = {'sparsity': self._sparsity()})

        return train_loss / len(self.train_loader)

    def _valid_epoch(self, epoch):

        valid_loss = 0.0
        for x, x_dat in self.valid_loader:
            valid_loss += self._loss(x, x_dat, self.model(x)).item()

        return valid_loss / len(self.valid_loader)

    def _update_best(self, epoch):

        valid_losses = self.history['valid_losses']
        curr_acc = valid_losses[-1]
        best_acc = valid_losses[self.best['epoch']-1]
        curr_spar = self._sparsity()
        best_spar = self.best.get('sparsity', 0)

        if curr_spar > best_spar and curr_acc < best_acc + .1:
            self.best.update({
                'epoch': epoch,
                'sparsity': curr_spar,
                'model_dict': deepcopy(self.model.state_dict()),
                'optim_dict': deepcopy(self.optimizer.state_dict())
            })


    def _prune(self):
        self.prune_func(**self.prune_args)
        self.sparsity = self._sparsity()

    def _parse_prune(self, args):
        select_params = [
            param_name.split('.')
            for param_name in args['parameters']
        ]
        named_modules = dict(chain(self.model.encoder.named_modules(),
                                   self.model.decoder.named_modules()))
        args['parameters'] = [
            (named_modules[name], param)
            for name, param in select_params
        ]
        args['pruning_method'] = getattr(prune, args['pruning_method'])

        return args

    def _loss(self, x, x_covi, x_model):
        x_rec, _ = x_model
        # compute sample local noise covariances of reconstructed points
        with torch.no_grad():
            sample = x_rec.detach().numpy()
            covs = lnc_ito(sample, self.dataset.sde)
            # covs = ln_covs(sample, self.sde, self.solver,
            #                config['burst_size'], config['burst_dt'])
            x_rec_covi = torch.pinverse(torch.as_tensor(covs), rcond=1e-10)

        return self.cost(x, x_rec, x_covi + x_rec_covi)

    def _sparsity(self):  # TODO: model attribute ?
        num = 0.0
        den = 0.0
        for module in chain(self.model.encoder, self.model.decoder):
            if hasattr(module, 'weight'):
                num += torch.sum(module.weight == 0)
                den += module.weight.nelement()
            if hasattr(module, 'bias') and module.bias is not None:
                num += torch.sum(module.bias == 0)
                den += module.bias.nelement()

        return float(num) / float(den)

    def _save(self, *args):
        # self.info['sparsity'] = self.sparsity
        super()._save(*args)
