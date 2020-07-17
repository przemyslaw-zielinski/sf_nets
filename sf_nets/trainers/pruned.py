#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 5 Jun 2020

@author: Przemyslaw Zielinski
"""

import torch
from itertools import chain
import torch.nn.utils.prune as prunelib
from utils.dmaps import ln_covs, lnc_ito

from copy import deepcopy
from .base import BaseTrainer

class PrunedTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.pruning = self.config['pruning']

        self.prune_func = getattr(prunelib, self.pruning['type'])
        self.prune_args = self._parse_prune(self.pruning['args'])

        self.start, self.freq = self.pruning['schedule']
        # self.sparsity = 0.0

    def _train_epoch(self, epoch):

        epoch_loss = 0.0
        for x, x_dat in self.train_loader:

            self.optimizer.zero_grad()
            batch_loss = self.compute_loss(x, x_dat, self.model(x))
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()

        if epoch >= self.start and epoch % self.freq == 0 and self.model.sparsity < self.pruning['target_sparsity']:
            self._prune()
            print(f'epoch : {epoch}, pruned!, sparsity = {self.model.sparsity:.2f}')

        # if epoch % 15 == 0:
        #     self._save_checkpoint(epoch, info = {'sparsity': model.sparsity})

        return epoch_loss / len(self.train_loader)

    def _valid_epoch(self, epoch):

        valid_loss = 0.0
        for x, x_dat in self.valid_loader:
            valid_loss += self.compute_loss(x, x_dat, self.model(x)).item()

        return valid_loss / len(self.valid_loader)

    def _update_best(self, epoch):

        valid_losses = self.history['valid_losses']
        curr_acc = valid_losses[-1]
        best_acc = valid_losses[self.best['epoch']-1]
        curr_spar = self.model.sparsity #self._sparsity()
        best_spar = self.best.get('sparsity', 0)

        if curr_spar >= self.pruning['target_sparsity'] and curr_acc < best_acc + .01:
            self.best.update({
                'epoch': epoch,
                'sparsity': curr_spar,
                'model_dict': deepcopy(self.model.state_dict()),
                'optim_dict': deepcopy(self.optimizer.state_dict())
            })


    def _prune(self):
        self.prune_func(**self.prune_args)

    def _parse_prune(self, args):
        selected_params = [
            param_name.rsplit('.', maxsplit=1)
            for param_name in args['parameters']
        ]
        named_modules = dict(self.model.named_modules())
        args['parameters'] = [
            (named_modules[name], param)
            for name, param in selected_params
        ]
        args['pruning_method'] = getattr(prunelib, args['pruning_method'])

        return args

    def compute_loss(self, x, x_covi, x_model):
        x_rec, _ = x_model
        # compute sample local noise covariances of reconstructed points
        with torch.no_grad():
            sample = x_rec.detach().numpy()
            covs = lnc_ito(sample, self.dataset.sde)
            # covs = ln_covs(sample, self.sde, self.solver,
            #                config['burst_size'], config['burst_dt'])
            x_rec_covi = torch.pinverse(torch.as_tensor(covs), rcond=1e-10)

        return self.loss(x, x_rec, x_covi + x_rec_covi)

    def _save(self, *args):
        # self.info['sparsity'] = self.sparsity
        super()._save(*args)
