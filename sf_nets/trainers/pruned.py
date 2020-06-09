#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 5 Jun 2020

@author: Przemyslaw Zielinski
"""

import torch
import torch.nn.utils.prune as prune
from utils.dmaps import ln_covs, lnc_ito

from .base import BaseTrainer

class PrunedTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        prune_list = []
        for name, module in self.model.encoder.named_modules():
            if name in self.config['layers_to_prune']:
                prune_list.append((module, 'weight'))
                prune_list.append((module, 'bias'))
        for name, module in self.model.decoder.named_modules():
            if name in self.config['layers_to_prune']:
                prune_list.append((module, 'weight'))
                prune_list.append((module, 'bias'))
        self.prune_list = prune_list
        print(self.prune_list)

    def _train_epoch(self, epoch):

        train_loss = 0.0
        for x, x_dat in self.train_loader:

            self.optimizer.zero_grad()
            loss = self._loss(x, x_dat, self.model(x))
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        amount = self.config['target_sparsity'] * epoch / self.config['max_epochs']
        if epoch > 10 and self._sparsity() < amount:
            prune.global_unstructured(
                self.prune_list,
                pruning_method=prune.L1Unstructured,
                amount=0.1,
            )
            # print(self.model.encoder)
            print(f'Pruned! Sparsity: {self._sparsity()}')

        return train_loss / len(self.train_loader)

    def _valid_epoch(self, epoch):

        valid_loss = 0.0
        for x, x_dat in self.valid_loader:
            valid_loss += self._loss(x, x_dat, self.model(x)).item()

        return valid_loss / len(self.valid_loader)

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

    def _sparsity(self):
        num = 0.0
        den = 0.0
        for module in self.model.encoder:
            # breakpoint()
            if hasattr(module, 'weight'):
                num += torch.sum(module.weight == 0)
                den += module.weight.nelement()
            if 'bias' in module.__dict__['_parameters']:
                if module.bias is not None:
                    num += torch.sum(module.bias == 0)
                    den += module.bias.nelement()
        for module in self.model.decoder:
            if 'weight'in module.__dict__['_parameters']:
                num += torch.sum(module.weight == 0)
                den += module.weight.nelement()
            if 'bias' in module.__dict__['_parameters']:
                if module.bias is not None:
                    num += torch.sum(module.bias == 0)
                    den += module.bias.nelement()
        # breakpoint()
        # print(num, den)

        return float(num) / float(den)
