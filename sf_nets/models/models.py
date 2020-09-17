#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 17 Sep 2020

@author: Przemyslaw Zielinski
"""

import torch
from nets import SimpleAutoencoder
from losses import MahalanobisLoss

class MahalanobisAutoencoder(SimpleAutoencoder):

    def __init__(self, system, input_features, latent_features, hidden_features=[]):

        super().__init__(input_features, latent_features, hidden_features=[])

        self.system = system
        self.mah_loss = MahalanobisLoss()

    def loss(self, batch):
        x, x_covi = batch
        x_rec = self(x)
        # compute sample local noise covariances of reconstructed points
        with torch.no_grad():
            x_rec_np = x_rec.detach().numpy()
            ln_covs = self.system.ln_covs(x_rec_np))
            x_rec_covi = torch.pinverse(torch.as_tensor(ln_covs), rcond=1e-10)

        return self.mah_loss(x, x_rec, (x_covi + x_rec_covi))
