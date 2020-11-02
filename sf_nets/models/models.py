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
from .nets import SimpleAutoencoder
# from .losses import MahalanobisLoss

class Autoencoder(SimpleAutoencoder):

    def __init__(self, *args, loss_fn='Sigmoid', **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = getattr(torch.nn, loss_fn)()

    def set_system(self, system):
        pass

    def loss(self, batch):
        x, *rest = batch
        x_rec, _ = self(x)

        return self.loss_fn(x, x_rec)

class MahalanobisAutoencoder(SimpleAutoencoder):

    def __init__(self, input_features, latent_features, hidden_features=[]):

        super().__init__(input_features, latent_features, hidden_features)
        self.loss_fn = losses.MahalanobisLoss()

    def set_system(self, system):
        self.system = system

    def loss(self, batch):
        x, x_covi, *rest = batch
        x_rec, _ = self(x)
        # compute sample local noise covariances of reconstructed points
        with torch.no_grad():
            x_rec_np = x_rec.detach().numpy()
            ln_covs = self.system.eval_lnc(x_rec_np, None, None, None)
            x_rec_covi = torch.pinverse(torch.as_tensor(ln_covs), rcond=1e-12)

        mah_loss = self.loss_fn(x, x_rec, x_covi + x_rec_covi)
        if torch.isnan(mah_loss).any():
            breakpoint()
        return mah_loss

class SemiMahalanobisAutoencoder(SimpleAutoencoder):

    def __init__(self, input_features, latent_features, hidden_features=[]):

        super().__init__(input_features, latent_features, hidden_features)
        self.mah_loss = losses.MahalanobisLoss()

    def set_system(self, system):
        self.system = system

    def loss(self, batch):
        x, x_covi, *rest = batch
        x_rec, _ = self(x)

        return self.mah_loss(x, x_rec, x_covi)

class MahalanobisL1Autoencoder(SimpleAutoencoder):

    def __init__(self, input_features, latent_features, hidden_features=[],
                 mah_weight=0.5, l1_weight=0.5):

        super().__init__(input_features, latent_features, hidden_features)
        self.mah_weight = mah_weight
        self.mah_loss = losses.MahalanobisLoss()
        self.l1_weight = l1_weight
        self.l1_loss = torch.nn.L1Loss()

    def set_system(self, system):
        self.system = system

    def loss(self, batch):
        x, x_covi, x_proj = batch
        x_rec, _ = self(x)
        # compute sample local noise covariances of reconstructed points
        with torch.no_grad():
            x_rec_np = x_rec.detach().numpy()
            ln_covs = self.system.eval_lnc(x_rec_np, None, None, None)
            x_rec_covi = torch.pinverse(torch.as_tensor(ln_covs), rcond=1e-10)

        # P = torch.zeros(4, 4)
        # P[3, 3] = 1.0  # projection on last coord
        # x_proj = x @ P

        return (
            self.mah_weight*self.mah_loss(x, x_rec, (x_covi + x_rec_covi)) +
            self.l1_weight*self.l1_loss(x_rec, x_proj)
            )

class MSEAutoencoder(SimpleAutoencoder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.MSELoss()

    def set_system(self, system):
        pass

    def loss(self, batch):
        x, *rest = batch
        x_rec, _ = self(x)

        return self.loss_fn(x, x_rec)

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
