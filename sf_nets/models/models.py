#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 17 Sep 2020

@author: Przemyslaw Zielinski
"""

import torch
from . import losses
import torch.nn as nn
import torch.nn.functional as F
from .nets import BaseAutoencoder
from collections import OrderedDict

class SimpleAutoencoder(BaseAutoencoder):

    def __init__(self, *args, loss_func='MSELoss', loss_weight=1.0,
                    data_pos=0, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(loss_func, str):
            loss_func = [loss_func]
            loss_weight = [loss_weight]

        self.losses = nn.ModuleDict({
            f'loss{n+1}': getattr(losses, loss_func)()
            for n, loss_func in enumerate(loss_func)
        })
        self.loss_weights = {
            f'weight{n+1}': weight
            for n, weight in enumerate(loss_weight)
        }

        self.pos = data_pos

    def __repr__(self):

        repr = super().__repr__()[:-2]  # gets rid of '\n)'
        repr += "\n"
        tab = " " * 2
        repr += tab + "(loss_weights): Dict(\n"
        for key, val in self.loss_weights.items():
            repr += tab * 2
            repr += f"({key}): {val}\n"
        repr += tab + ")"
        repr += "\n)"
        return repr

    def set_system(self, system):
        pass

    def loss(self, batch):  # TODO: maybe 'eval_loss'?
        x = batch[self.pos]
        x_rec = self(x)

        loss_val = 0.0
        weights = self.loss_weights.values()
        loss_funcs = self.losses.values()
        for weight, loss_func in zip(weights, loss_funcs):
            loss_val += weight * loss_func(x, x_rec)

        return loss_val

class MahalanobisAutoencoder(BaseAutoencoder):

    def __init__(self, *base_args,
                 proj_loss="", proj_loss_wght=None, **base_kwargs):

        super().__init__(*base_args, **base_kwargs)

        self.args_dict.update({
            'proj_loss': proj_loss,
            'proj_loss_wght': proj_loss_wght
        })

        self.mah_loss = losses.MahalanobisLoss()
        self.mah_wght = 1.0

        self.proj_loss = getattr(losses, proj_loss, None)
        if self.proj_loss:
            self.proj_loss = self.proj_loss()
            self.proj_loss_wght = proj_loss_wght
            self.mah_wght -= proj_loss_wght

        self.metrics = {
            'tot_runs': 0,
            'proj_loss': 0.0,
            'proj_mse_loss': 0.0,
            'proj_max_loss': torch.Tensor([0.0])}

    def set_system(self, system):
        self.system = system

    def loss(self, batch):

        x, x_covi, x_proj = batch
        x_rec = self(x)

        # compute sample local noise covariances of reconstructed points
        with torch.no_grad():
            x_rec_np = x_rec.detach().numpy()
            ln_covs = self.system.eval_lnc(x_rec_np, None, None, None)
            x_rec_covi = torch.pinverse(torch.as_tensor(ln_covs), rcond=1e-12)
        loss_val = self.mah_wght * self.mah_loss(x, x_rec, x_covi + x_rec_covi)

        if self.proj_loss:
            loss_val += self.proj_loss_wght * self.proj_loss(x_rec, x_proj)

        return loss_val

    def update_metrics(self, batch):
        x, x_covi, x_proj = batch
        x_rec = self(x)

        self.metrics['tot_runs'] += 1
        self.metrics['proj_loss'] += self.proj_loss(x_rec, x_proj)
        self.metrics['proj_mse_loss'] += F.mse_loss(x_rec, x_proj)

        curr_max = self.metrics['proj_max_loss']
        batch_max = torch.max(torch.abs(x_rec - x_proj))
        self.metrics['proj_max_loss'] = torch.max(batch_max, curr_max)

    def compute_metrics(self):
        proj_loss = self.metrics['proj_loss'] / self.metrics['tot_runs']
        proj_mse_loss = self.metrics['proj_mse_loss'] / self.metrics['tot_runs']
        return {
            'proj_loss': proj_loss,
            'proj_mse_loss': proj_mse_loss,
            'proj_max_loss': self.metrics['proj_max_loss']
            }

    def reset_metrics(self):
        self.metrics['tot_runs'] = 0
        self.metrics['proj_loss'] = 0.0
        self.metrics['proj_mse_loss'] = 0.0
        self.metrics['proj_max_loss'] = torch.Tensor([0.0])

class SemiMahalanobisAutoencoder(BaseAutoencoder):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.mah_loss = losses.MahalanobisLoss()

    def set_system(self, system):
        self.system = system

    def loss(self, batch):
        x, x_covi, *rest = batch
        x_rec = self(x)

        return self.mah_loss(x, x_rec, x_covi)
