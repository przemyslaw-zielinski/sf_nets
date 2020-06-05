#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 13 May 2020

@author: Przemyslaw Zielinski
"""

import os
import torch
import torch.optim as optim
from copy import deepcopy
from .dmaps import ln_covs, lnc_ito
from collections import namedtuple
from torch.utils.data import DataLoader
from .nets import SimpleAutoencoder, MahalanobisLoss

Data = namedtuple('Data', ['train', 'valid'])

class Model():
    pass

class Simple():

    def __init__(self, params):
        self.model = SimpleAutoencoder(params['net_arch'])
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=params['learning_rate'])
        self.loss_func = MahalanobisLoss()

        self.sde = params.pop('sde')
        self.solver = params.pop('solver')
        self.params = params

        self.history = {
            'train_losses': [],
            'valid_losses': []
        }

        self.state = { # best fit
            'best_epoch': 1,
            'best_model_dict': {},
            'best_optim_dict': {},
            'last_model_dict': {},
            'last_optim_dict': {}
        }

    def loss(self, x, x_covi, x_model):
        x_rec, _ = x_model
        # compute sample local noise covariances of reconstructed points
        with torch.no_grad():
            sample = x_rec.detach().numpy()
            covs = lnc_ito(sample, self.sde)
            # covs = ln_covs(sample, self.sde, self.solver,
            #                params['burst_size'], params['burst_dt'])
            x_rec_covi = torch.pinverse(torch.as_tensor(covs), rcond=1e-10)

        return self.loss_func(x, x_rec, x_covi + x_rec_covi)

    def step(self, train_loader, valid_loader):
        train_loss = 0.0
        for x, x_dat in train_loader:
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            self.optimizer.zero_grad()
            # compute training reconstruction loss
            loss = self.loss(x, x_dat, self.model(x))
            # compute accumulated gradients
            loss.backward()
            # perform parameter update based on current gradients
            self.optimizer.step()
            # add the mini-batch training loss to epoch loss
            train_loss += loss.item()

        # compute the epoch validation loss
        with torch.no_grad():
            valid_loss = 0.0
            for x, x_dat in valid_loader:
                valid_loss += self.loss(x, x_dat, self.model(x)).item()

        return train_loss / len(train_loader), valid_loss / len(valid_loader)

    def _update_best(self, epoch):
        # update best fit based on validation performance
        curr_acc = self.history['valid_losses'][-1]
        best_acc = self.history['valid_losses'][self.state['best_epoch']-1]
        if epoch > 1 and curr_acc < best_acc:
            self.state['best_epoch'] = epoch
            self.state['best_model_dict'] = deepcopy(self.model.state_dict())
            self.state['best_optim_dict'] = deepcopy(self.optimizer.state_dict())

    def save(self):
        system = self.params['system']
        model_id = self.params['model_id']
        path = f'../models/{system}'
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({'params': self.params,
                    'history': self.history,
                    'state': self.state},
                   f'{path}/{model_id}.pt')

    def train(self, train_data, valid_data):

        train_loader = DataLoader(train_data,
                                  batch_size=self.params['batch_size'],
                                  shuffle=True)
        valid_loader = DataLoader(valid_data,
                                  batch_size=self.params['batch_size'],
                                  shuffle=True)

        for epoch in range(1, self.params['max_epochs'] + 1):
            train_loss, valid_loss = self.step(train_loader, valid_loader)
            self.history['train_losses'].append(train_loss)
            self.history['valid_losses'].append(valid_loss)
            self._update_best(epoch)
            # display the epoch loss
            if epoch == 1 or epoch % 10 == 0:
                print(f"epoch : {epoch:3d}/{self.params['max_epochs']}, "
                      f"reconstruction loss = {train_loss:.5f}, "
                      f"validation loss = {valid_loss:.5f}")

        self.state['last_model_dict'] = self.model.state_dict()
        self.state['last_optim_dict'] = self.optimizer.state_dict()
        self.save()

def train_simple(train_data, valid_data, params):

    data = Data(train=train_data, valid=valid_data)
    model = SimpleAutoencoder(params['net_arch'])

    m_loss = MahalanobisLoss()
    def loss(x, x_covi, x_model):

        x_rec, _ = x_model
        # compute sample local noise covariances of reconstructed points
        with torch.no_grad():
            sample = x_rec.detach().numpy()
            covs = lnc_ito(sample, params['sde'])
            # covs = ln_covs(sample, params['sde'], params['solver'],
            #                params['burst_size'], params['burst_dt'])
            x_rec_covi = torch.pinverse(torch.as_tensor(covs), rcond=1e-10)

        return m_loss(x, x_rec, x_covi + x_rec_covi)

    train_loop(data, model, loss, params)


def train_loop(data, model, loss, params):
    """
    Trains the model based on given data and loss function, and stores
    the results in a file specified with a path
        models/params['system']/params['model_id'].pt

    Parameters
    ----------
    data   : with fields: data.train, data.valid that subclass torch Dataset
             iteration over them returns tuple x, x_dat where
             -> x : the input variable
             -> x_dat : additional data to evaluate loss
    model  : torch nn.Module subclass that evaluates as module(x)
    loss   : evaluates loss based on x, x_dat and value returned by model(x)
    params : dictionary of additional params to store, required keys are:
             -> batch_size
             -> learning_rate
             -> max_epochs
             -> model_id
    """

    train_loader = DataLoader(data.train,
                              batch_size=params['batch_size'], shuffle=True)
    valid_loader = DataLoader(data.valid,
                              batch_size=params['batch_size'], shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    train_losses, valid_losses = [], []
    for epoch in range(1, params['max_epochs'] + 1):
        epoch_loss = 0.0
        for x, x_dat in train_loader:

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute training reconstruction loss
            train_loss = loss(x, x_dat, model(x))

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            epoch_loss += train_loss.item()

        # compute the epoch training loss
        train_losses.append(epoch_loss / len(train_loader))

        # compute the epoch validation loss
        with torch.no_grad():
            valid_loss = 0.0
            for x, x_dat in valid_loader:
                valid_loss += loss(x, x_dat, model(x)).item()
            valid_losses.append(valid_loss / len(valid_loader))

        # update best fit based on validation performance
        if epoch == 1:
            best_acc = valid_losses[-1]
        elif valid_losses[-1] < best_acc:
            best_model_dict = deepcopy(model.state_dict())
            best_optim_dict = deepcopy(optimizer.state_dict())
            best_epoch = epoch
            best_acc = valid_losses[-1]

        # display the epoch loss
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch : {epoch:3d}/{params['max_epochs']}, "
                  f"reconstruction loss = {train_losses[-1]:.5f}, "
                  f"validation loss = {valid_losses[-1]:.5f}")
    history = {
        'train_losses': train_losses,
        'valid_losses': valid_losses
    }

    state = { # best fit
        'best_epoch': best_epoch,
        'best_model_dict': best_model_dict,
        'best_optim_dict': best_optim_dict,
        'last_model_dict': model.state_dict(),
        'last_optim_dict': optimizer.state_dict()
    }

    system = params['system']
    model_id = params['model_id']
    path = f'../models/{system}'
    if not os.path.exists(path):
        os.makedirs(path)
    del params['sde']
    del params['solver']
    torch.save({'params': params, 'history': history, 'state': state},
               f'{path}/{model_id}.pt')
