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

# class classproperty(object):
#     def __init__(self, f):
#         self.f = f
#     def __get__(self, obj, owner):
#         return self.f(owner)

class RQP4System():

    nmd = 4
    ndim = 4
    sdim = 1

    @staticmethod
    def slow_map(x):
        return x[3] - x[2]**2 - x[1]**2

    @staticmethod
    def _sde_drift(tsep, temp):

        def drift(t, x, dx):
            dx[0] = -x[0] / tsep
            dx[1] = (x[0] - x[1]) / tsep
            dx[2] = (x[1] - x[2]) / tsep
            dx[3] = x[2] - x[3] + x[2]**2 + x[1]**2\
                  + 2*x[2]*(x[1]-x[2])/tsep + 2*x[1]*(x[0]-x[1])/tsep + 4*temp/tsep

        return drift

    @staticmethod
    def _sde_dispersion(tsep, temp):

        def disp(t, x, dx):
            fast_disp = np.sqrt(2*temp/tsep)
            slow_disp = np.sqrt(2*temp)

            dx[0,0] = fast_disp
            dx[1,1] = fast_disp
            dx[2,2] = fast_disp
            dx[3,1] = 2*x[1]*fast_disp
            dx[3,2] = 2*x[2]*fast_disp
            dx[3,3] = slow_disp

        return disp

    def __init__(self, tsep, temp):

        self.sde = spaths.ItoSDE(self._sde_drift(tsep, temp),
                                 self._sde_dispersion(tsep, temp),
                                 noise_mixing_dim=self.nmd)

    def eval_lnc(self, data, solver, burst_size, burst_dt, nsteps=1):
        """
        Computes local noise covaraiances for all instances in data.
        """

        if isinstance(self.sde, spaths.ItoSDE):
            disp_val = self.sde.ens_disp(0, data)
            return np.einsum('bij,bkj->bik', disp_val, disp_val)
        else:
            data_rep = np.repeat(data.astype(dtype=np.float32), burst_size, axis=0)
            batch = solver.burst(self.sde, data_rep, (0.0, nsteps), burst_dt)

            covs = []
            fact = nsam - 1
            for point_batch in np.split(batch, len(data)):
                point_batch -= np.average(point_batch, axis=0)
                covs.append(point_batch.T @ point_batch / (dt * fact))

            return np.array(covs)

class RQP4(Dataset):

    # system parameters
    tsep = 0.04
    temp = 0.05

    # system
    system = RQP4System(tsep, temp)

    # simulation parameters
    seed = 3579
    dt = .1 * tsep
    x0 = [0.0, 0.0, 0.0, 0.0]
    tspan = (0.0, 160)

    # lnc computations
    burst_size = 10**4
    burst_dt = dt

    # data files
    train_file = 'train.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, generate=False, transform=None):

        self.root = Path(root)

        if generate:

            (self.root / self.name).mkdir(exist_ok=True)
            self.raw.mkdir(exist_ok=True)
            self.processed.mkdir(exist_ok=True)

            solution, train_ds, test_ds = self.generate()

            torch.save((solution.t, solution.p[0]), self.raw / 'path.pt')
            torch.save(train_ds, self.processed / self.train_file)
            torch.save(test_ds, self.processed / self.test_file)

        if not self._check_exists():
            raise RuntimeError('Dataset not found! '
                               'Use generate=True to generate it.')
        if train:
            data_file = self.train_file
        else:
            data_file = self.test_file
        self.data, self.ln_covs = torch.load(self.processed / data_file)

    @property
    def name(self):
        return type(self).__name__

    @property
    def processed(self):
        return self.root / self.name / 'processed'

    @property
    def raw(self):
        return self.root / self.name / 'raw'

    def __repr__(self):
        head = 'Dataset ' + self.name
        body = [f'Number of datapoints: {len(self)}']

        indent = ' ' * 4
        lines = [head] + [indent + line for line in body]

        return '\n'.join(lines)

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
        # TODO: store and check the simulation metadata
        return (
            (self.processed/self.train_file).exists() and
            (self.processed/self.test_file).exists()
            )

    @classmethod
    def generate(cls):
        '''
        Returns
        -------
        solution : an instance of spath class that stores the full simulation data
        train_ds : a tuple with train datapoints and corr. inverse local
                   noise covariance matrices
        test_ds : a tuple with test datapoints and corr. inverse local
                  noise covariance matrices
        '''

        # print(f'Generating {cls.__class__.__name__} dataset.')

        # seed setting
        rng = np.random.default_rng(cls.seed)
        rng.integers(10**3);  # warm up of RNG

        # solver
        em = spaths.EulerMaruyama(rng)

        sol =  em.solve(cls.system.sde, np.array([cls.x0]), cls.tspan, cls.dt)
        path = sol.p[0]

        # skip first few samples and from the rest take only a third
        t_data = sol.t[100::3]
        data = np.squeeze(sol(t_data)).astype(dtype=np.float32)

        # compute local noise covariances at data points
        covs = cls.system.eval_lnc(data, em, cls.burst_size, cls.burst_dt)

        data_t = torch.from_numpy(data).float()
        # TODO: store covariances and use transform parameter to invert while
        #       reading the data
        covi_t = torch.pinverse(torch.tensor(covs).float(), rcond=1e-10)

        data_train, data_test, covi_train, covi_test = train_test_split(
            data_t, covi_t, test_size=0.33, random_state=rng.integers(1e5)
        )

        return sol, (data_train, covi_train), (data_test, covi_test)
