#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 19 Oct 2020

@author: Przemyslaw Zielinski
"""

import os
import torch
import warnings
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from .base import SimDataset
import utils.dmaps as dmaps
import utils.spaths as spaths
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class LogSin2System():

    nmd = 2
    ndim = 2
    sdim = 1

    # helper functions
    f = lambda self, r: jnp.log(1 + r**2)
    g = lambda self, r: jnp.sin(.5 * r)
    ginv = lambda self, s: 2 * jnp.arcsin(s)

    @staticmethod
    def slow_map(x):
        f = lambda r: np.log(1 + r**2)
        ginv = lambda s: 2 * np.arcsin(s)
        G = lambda x, y: np.array([ginv(y-f(x)) + x, x])
        return G(*x)[0]

    @staticmethod
    def _lin_drift(tsep):

        def drift(t, u, du):
            du[0] = 1.0
            du[1] = (u[0] - u[1]) / tsep

        return drift

    @staticmethod
    def _lin_dispersion(tsep):

        def disp(t, u, du):
            du[0,0] = 1.0 / np.sqrt(100.0)
            du[1,1] = 1.0 / np.sqrt(3*tsep)

        return disp

    def __init__(self, tsep):

        # in coordinates
        self.F = lambda u, v: jnp.array([v, self.f(v) + self.g(u-v)])
        self.G = lambda x, y: jnp.array([self.ginv(y-self.f(x)) + x, x])

        # for data arrays
        fwdF = lambda uv: self.F(uv[0], uv[1])
        bwdF = lambda xy: self.G(xy[0], xy[1])


        sde_ou = spaths.ItoSDE(self._lin_drift(tsep),
                               self._lin_dispersion(tsep),
                               noise_mixing_dim=self.nmd)
        transform = spaths.SDETransform(fwdF, bwdF)
        self.sde = transform(sde_ou)

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

class LogSin2(SimDataset):

    # system parameters
    tsep = 0.001

    # system
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        system = LogSin2System(tsep)
        x0 = system.F(3.0, 3.0)

    # simulation parameters
    seed = 3579
    dt = tsep / 5
    tspan = (0.0, 8.0)

    # lnc computations
    burst_size = 10**4
    burst_dt = dt

    def load(self, data_path):
        self.data, self.ln_covs = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx], self.ln_covs[idx]

    @classmethod
    def generate(cls):
        '''
        Returns
        -------
        solution : a tuple of times and path values
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
        t_data = sol.t[1::4]
        data = np.squeeze(sol(t_data)).astype(dtype=np.float32)

        # compute local noise covariances at data points
        covs = cls.system.eval_lnc(data, em, cls.burst_size, cls.burst_dt)

        data_t = torch.from_numpy(data).float()
        # TODO: store covariances and use transform parameter to invert while
        #       reading the data
        covi_t = torch.pinverse(torch.tensor(covs).float(), rcond=1e-10)

        data_train, data_test, covi_train, covi_test = train_test_split(
            data_t, covi_t, test_size=0.33, random_state=rng.integers(1e3)
        )

        return (sol.t, path), (data_train, covi_train), (data_test, covi_test)
