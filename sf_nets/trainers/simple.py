#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 5 Jun 2020

@author: Przemyslaw Zielinski
"""

import torch
from utils.dmaps import ln_covs, lnc_ito

from .base import BaseTrainer

class SimpleTrainer(BaseTrainer):

    def __init_subclass__(cls):
        super.__init_subclass__()

    def _train_epoch(self, epoch):

        epoch_loss = 0.0
        for x, x_dat in self.train_loader:
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            self.optimizer.zero_grad()
            # compute training reconstruction loss
            # TODO: maybe: compute_loss(x, self.model(x), *x_dat)
            batch_loss = self.compute_loss(x, x_dat, self.model(x))
            # compute accumulated gradients
            batch_loss.backward()
            # perform parameter update based on current gradients
            self.optimizer.step()
            # add the mini-batch training loss to epoch loss
            epoch_loss += batch_loss.item()

        return epoch_loss / len(self.train_loader)

    def _valid_epoch(self, epoch):

        valid_loss = 0.0
        for x, x_dat in self.valid_loader:
            valid_loss += self.compute_loss(x, x_dat, self.model(x)).item()

        return valid_loss / len(self.valid_loader)

    def compute_loss(self, x, x_covi, x_model):
        x_rec, _ = x_model
        # compute sample local noise covariances of reconstructed points
        with torch.no_grad():
            sample = x_rec.detach().numpy()
            covs = lnc_ito(sample, self.dataset.sde)
            # covs = ln_covs(sample, self.sde, self.solver,
            #                config['burst_size'], config['burst_dt'])
            x_rec_covi = torch.pinverse(torch.as_tensor(covs), rcond=1e-10)

        # e_vals = torch.symeig(x_covi).eigenvalues[:, -1]  # only 1 dominant e-val

        return self.loss(x, x_rec, (x_covi + x_rec_covi))# / e_vals[:,None, None])

class MMSELossTrainer(SimpleTrainer):

    def compute_loss(self, x, x_covi, x_model):
        x_rec, _ = x_model
        # compute sample local noise covariances of reconstructed points
        with torch.no_grad():
            sample = x_rec.detach().numpy()
            covs = lnc_ito(sample, self.dataset.sde)
            # covs = ln_covs(sample, self.sde, self.solver,
            #                config['burst_size'], config['burst_dt'])
            x_rec_covi = torch.pinverse(torch.as_tensor(covs), rcond=1e-10)
            x_rec_drif = torch.as_tensor(self.dataset.sde.drif(0, sample))
            x_rec_mean = x_rec + x_rec_drif * self.dataset.burst_dt

        mse_loss = torch.nn.MSELoss()
        return (
            # .5*self.loss(x, x_rec, (x_covi + x_rec_covi)) +
            mse_loss(x_rec, x_rec_mean) / (self.dataset.burst_dt**3)
            )

class SimpleLossTrainer(SimpleTrainer):

    def compute_loss(self, x, x_covi, x_model):
        x_rec, _ = x_model
        return self.loss(x, x_rec)
