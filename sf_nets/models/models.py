#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 17 Sep 2020

@author: Przemyslaw Zielinski
"""

import torch
from . import losses
from .nets import SimpleAutoencoder
# from .losses import MahalanobisLoss

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
