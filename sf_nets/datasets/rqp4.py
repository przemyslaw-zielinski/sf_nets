#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 5 Jun 2020

@author: Przemyslaw Zielinski
"""

import os
import torch
import spaths
import numpy as np
from torch.utils.data import Dataset

class RQP4(Dataset):

    name = 'RQP4'
    ndim = 4
    sdim = 1
    nmd = 4
    tsep = 0.04
    temp = 0.05
    seed = 3579
    # data = {
    #     'ndim': 4,
    #     'sdim': 1,
    #     'nmd': 4,
    #     'tsep': 0.04,
    #     'temp': 0.05,
    # }
    # TODO: add simulation parameters
    train_file = 'train.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, simulate=False):

        self.root = root
        self.train = train  # training set or test set

        if simulate:
            self.simulate()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. '
                               'You can use simulate=True to generate it')
        if self.train:
            data_file = self.train_file
        else:
            data_file = self.test_file
        # TODO: add raw and processed folders
        self.data, self.ln_covs = torch.load(os.path.join(self.root,
                                                          self.__class__.__name__,
                                                          data_file))

        self.sde = spaths.ItoSDE(self._sde_drift, self._sde_dispersion,
                                 noise_mixing_dim=self.nmd)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (point, ln_cov) where ln_cov is the local noise covariance
                   associated to the data point.
        """
        return self.data[idx], self.ln_covs[idx]

    def _check_exists(self):
        # TODO: check the simulation metadata
        return (os.path.exists(os.path.join(self.root, self.__class__.__name__,
                                            self.train_file)) and
                os.path.exists(os.path.join(self.root, self.__class__.__name__,
                                            self.test_file)))

    def simulate(self):
        pass
        # TODO:

    def _sde_drift(self, t, x, dx):
        tsep = self.tsep
        temp = self.temp

        dx[0] = -x[0] / tsep
        dx[1] = (x[0] - x[1]) / tsep
        dx[2] = (x[1] - x[2]) / tsep
        dx[3] = x[2] - x[3] + x[2]**2 + x[1]**2\
              + 2*x[2]*(x[1]-x[2])/tsep + 2*x[1]*(x[0]-x[1])/tsep + 4*temp/tsep

    def _sde_dispersion(self, t, x, dx):
        fast_disp = np.sqrt(2*self.temp/self.tsep)
        slow_disp = np.sqrt(2*self.temp)

        dx[0,0] = fast_disp
        dx[1,1] = fast_disp
        dx[2,2] = fast_disp
        dx[3,1] = 2*x[1]*fast_disp
        dx[3,2] = 2*x[2]*fast_disp
        dx[3,3] = slow_disp

    def slow_map(x):
        return x[3] - x[2]**2 - x[1]**2
