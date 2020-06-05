#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 5 Jun 2020

@author: Przemyslaw Zielinski
"""

class SimpleTrainer():

    def __init__(self, model, loss_func, optimizer, config):
        self.model = SimpleAutoencoder(config['net_arch'])
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=config['learning_rate'])
        self.loss_func = MahalanobisLoss()

        self.sde = config.pop('sde')
        self.solver = config.pop('solver')
        self.config = config

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
            #                config['burst_size'], config['burst_dt'])
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
        system = self.config['system']
        model_id = self.config['model_id']
        path = f'../models/{system}'
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({'config': self.config,
                    'history': self.history,
                    'state': self.state},
                   f'{path}/{model_id}.pt')

    def train(self, train_data, valid_data):

        train_loader = DataLoader(train_data,
                                  batch_size=self.config['batch_size'],
                                  shuffle=True)
        valid_loader = DataLoader(valid_data,
                                  batch_size=self.config['batch_size'],
                                  shuffle=True)

        for epoch in range(1, self.config['max_epochs'] + 1):
            train_loss, valid_loss = self.step(train_loader, valid_loader)
            self.history['train_losses'].append(train_loss)
            self.history['valid_losses'].append(valid_loss)
            self._update_best(epoch)
            # display the epoch loss
            if epoch == 1 or epoch % 10 == 0:
                print(f"epoch : {epoch:3d}/{self.config['max_epochs']}, "
                      f"reconstruction loss = {train_loss:.5f}, "
                      f"validation loss = {valid_loss:.5f}")

        self.state['last_model_dict'] = self.model.state_dict()
        self.state['last_optim_dict'] = self.optimizer.state_dict()
        self.save()
