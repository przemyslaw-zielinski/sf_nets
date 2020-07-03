#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 5 Jun 2020

@author: Przemyslaw Zielinski
"""

import os
import torch
import numpy as np
from pathlib import Path
import utils.dmaps as dmaps
import utils.spaths as spaths
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class RQP4(Dataset):

    # system parameters
    ndim = 4
    sdim = 1
    nmd = 4
    tsep = 0.04
    temp = 0.05

    # simulation parameters
    seed = 3579
    dt = .1 * tsep
    x0 = [0.0, 0.0, 0.0, 0.0]
    tspan = (0.0, 160)
    burst_size = 10**4
    burst_dt = dt

    # files
    train_file = 'train.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, generate=False):

        # self.root = Path(root)
        self.raw = Path(root) / Path(f'{self.name}/raw')
        self.processed = Path(root) / Path(f'{self.name}/processed')

        self.raw.mkdir(exist_ok=True)
        self.processed.mkdir(exist_ok=True)
        # self.train = train  # training set or test set

        if generate:
            self.generate_data()

        if not self._check_exists():
            raise RuntimeError('Dataset not found! '
                               'Use generate=True to generate it.')
        if train:
            data_file = self.train_file
        else:
            data_file = self.test_file
        # TODO: add raw and processed folders
        self.data, self.ln_covs = torch.load(self.processed/data_file)

    @property
    def name(self):
        return type(self).__name__

    @property
    def sde(self):
        return spaths.ItoSDE(self._sde_drift, self._sde_dispersion,
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
        return (
            (self.processed/self.train_file).exists() and
            (self.processed/self.test_file).exists()
            )

    def generate_data(self):

        print(f'Generating {self.__class__.__name__} dataset.')

        # seed setting
        rng = np.random.default_rng(self.seed)
        rng.integers(10**3, size=10**4);  # warm up of RNG

        # solver
        em = spaths.EulerMaruyama(rng)

        sol =  em.solve(self.sde, np.array([self.x0]), self.tspan, self.dt)
        path = sol.p[0]
        torch.save(path, self.raw/'path.pt')

        # skip first few samples and from the rest take only a third
        t_data = sol.t[100::3]
        data = np.squeeze(sol(t_data)).astype(dtype=np.float32)

        # compute local noise covariances at data points
        covs = dmaps.ln_covs(data, self.sde, em, self.burst_size, self.burst_dt)

        data_t = torch.from_numpy(data).float()
        covi_t = torch.pinverse(torch.tensor(covs).float(), rcond=1e-10)

        data_train, data_test, covi_train, covi_test = train_test_split(
            data_t, covi_t, test_size=0.33, random_state=rng.integers(1e5)
        )

        torch.save((data_train, covi_train), self.processed/self.train_file)
        torch.save((data_test, covi_test), self.processed/self.test_file)


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
