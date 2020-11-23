#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 17 Sep 2020

@author: Przemyslaw Zielinski
"""

import torch
from . import losses
import torch.nn as nn
# import pytorch_lightning as pl
from collections import OrderedDict
from .nets import BaseAutoencoder

class SimpleAutoencoder(BaseAutoencoder):

    def __init__(self, *args, loss_func='MSELoss', loss_weight=1.0, **kwargs):
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
        x, *rest = batch
        x_rec = self(x)

        loss_val = 0.0
        weights = self.loss_weights.values()
        loss_funcs = self.losses.values()
        for weight, loss_func in zip(weights, loss_funcs):
            loss_val += weight * loss_func(x, x_rec)

        return loss_val

class MahalanobisAutoencoder(SimpleAutoencoder):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs, loss_func="MahalanobisLoss")

    def set_system(self, system):
        self.system = system

    def loss(self, batch):
        x, x_covi, *rest = batch
        x_rec = self(x)
        # compute sample local noise covariances of reconstructed points
        with torch.no_grad():
            x_rec_np = x_rec.detach().numpy()
            ln_covs = self.system.eval_lnc(x_rec_np, None, None, None)
            x_rec_covi = torch.pinverse(torch.as_tensor(ln_covs), rcond=1e-12)

        mah_loss = self.losses['loss1']
        # if torch.isnan(mah_loss).any():
        #     breakpoint()
        return mah_loss(x, x_rec, x_covi + x_rec_covi)

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

class MahalanobisL1Autoencoder(SimpleAutoencoder):

    def __init__(self, *args, mah_weight=0.5, l1_weight=0.5, **kwargs):

        loss_func = ["MahalanobisLoss", "L1Loss"]
        loss_weight = [mah_weight, l1_weight]

        super().__init__(*args, loss_func=loss_func, loss_weight=loss_weight, **kwargs)

    def set_system(self, system):
        self.system = system

    def loss(self, batch):
        x, x_covi, x_proj = batch
        x_rec = self(x)
        # compute sample local noise covariances of reconstructed points
        with torch.no_grad():
            x_rec_np = x_rec.detach().numpy()
            ln_covs = self.system.eval_lnc(x_rec_np, None, None, None)
            x_rec_covi = torch.pinverse(torch.as_tensor(ln_covs), rcond=1e-10)

        mw = self.loss_weights['weight1']
        lw = self.loss_weights['weight2']
        return (
            mw * self.losses['loss1'](x, x_rec, (x_covi + x_rec_covi)) +
            lw * self.losses['loss2'](x_rec, x_proj)
            )

# class LightAutoencoder(pl.LightningModule):
#
#     def __init__(self, inp_dim, lat_dim, hid_dims=[],
#                        hid_act=nn.ReLU(), lat_act=None, out_act=None):
#
#         super().__init__()
#
#         encoder = OrderedDict()
#         decoder = OrderedDict()
#
#         n = 0  # in case of no hidden layers
#         s = 2 + len(hid_dims)
#         if out_act:
#             decoder[f'activation{s-n-1}'] = out_act
#         for n, dim in enumerate(hid_dims):
#             n += 1
#             encoder[f'layer{n}'] = nn.Linear(inp_dim, dim)
#             encoder[f'activation{n}'] = hid_act
#             decoder[f'layer{s-n}'] = nn.Linear(dim, inp_dim)
#             decoder[f'activation{s-n-1}'] = hid_act
#             inp_dim = dim
#
#         n += 1
#         encoder[f'layer{n}'] = nn.Linear(inp_dim, lat_dim)
#         if lat_act:
#             encoder[f'activation{n}'] = lat_act
#         decoder[f'layer{s-n}'] = nn.Linear(lat_dim, inp_dim)
#
#         decoder = OrderedDict(reversed(decoder.items()))
#
#         self.encoder = nn.Sequential(encoder)
#         self.decoder = nn.Sequential(decoder)
#
#     def forward(self, x):
#         # in lightning, forward defines the prediction/inference actions
#         return self.encoder(x)
#
#     def training_step(self, batch, batch_idx):
#         # training_step defines the train loop. It is independent of forward
#         x, y = batch
#         z = self.encoder(x)
#         x_hat = self.decoder(z)
#         loss = nn.functional.mse_loss(x_hat, x)
# #         self.log('mse_loss', loss, on_epoch=True)
#         logs = {'loss': loss}
#
# #         result = pl.TrainResult(loss)
#         return {'loss': loss, 'log': logs}
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer
