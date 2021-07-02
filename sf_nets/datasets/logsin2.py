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
import sf_nets.utils.dmaps as dmaps
import spaths
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class LogSin2System():

    nmd = 2
    ndim = 2
    sdim = 1

    # helper functions
    f = lambda self, r: jnp.log(1 + r**2)
    g = lambda self, r: r #jnp.sin(.5 * r)
    ginv = lambda self, s: s #2 * jnp.arcsin(s)

    @staticmethod
    def slow_map(x):
        f = lambda r: np.log(1 + r**2)
        ginv = lambda s: s #2 * np.arcsin(s)
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

        # underlying linear system
        sde_lin = spaths.ItoSDE(self._lin_drift(tsep),
                               self._lin_dispersion(tsep),
                               noise_mixing_dim=self.nmd)

        # transform in in coordinates
        self.F = lambda u, v: jnp.array([v, self.f(v) + self.g(u-v)])
        self.G = lambda x, y: jnp.array([self.ginv(y-self.f(x)) + x, x])

        # transform for data arrays
        fwdF = lambda uv: self.F(uv[0], uv[1])
        bwdF = lambda xy: self.G(xy[0], xy[1])

        # # Z normalization
        # fwdZ = lambda x: (x - means) / stds
        # bwdZ = lambda y: stds * (y + means)

        Ftransform = spaths.SDETransform(fwdF, bwdF)

        # ZFtransform = spaths.SDETransform(lambda u: fwdZ(fwdF(u)),
        #                                   lambda y: bwdF(bwdZ(y)))
        self.sde = Ftransform(sde_lin)

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
        # self.data, self.ln_covs = torch.load(data_path)
        data, ln_covs, slow_proj = torch.load(data_path)

        data_np = data.detach().numpy()
        slow_proj_np = slow_proj.detach().numpy()
        means = jnp.mean(data_np, axis=0)
        stds = jnp.std(data_np, axis=0)

        fwdZ = lambda x: jnp.array([(x[0] - means[0]) / stds[0], (x[1] - means[1]) / stds[1]]) #(x - means) / stds
        bwdZ = lambda y: jnp.array([stds[0] * y[0] + means[0], stds[1] * y[1] + means[1]])

        Zdata_np = np.array(fwdZ(data_np.T).T)
        Zslow_proj_np = np.array(fwdZ(slow_proj_np.T).T)
        self.data = torch.from_numpy(Zdata_np)
        self.slow_proj = torch.from_numpy(Zslow_proj_np)

        # self.ln_covs = ln_covs

        transformZ = spaths.SDETransform(fwdZ, bwdZ)
        self.system = LogSin2System(self.tsep)
        old_slow_map = self.system.slow_map
        self.system.sde = transformZ(self.system.sde)
        self.system.slow_map = lambda y: old_slow_map(bwdZ(y))
        # print(self.system.sde.ens_disp(0, Zdata_np[:10]))

        # seed setting
        rng = np.random.default_rng(self.seed)
        rng.integers(10**3);  # warm up of RNG

        # solver
        em = spaths.EulerMaruyama(rng)

        # compute local noise covariances at data points
        covs = self.system.eval_lnc(Zdata_np, em, self.burst_size, self.burst_dt)
        self.ln_covs = torch.pinverse(torch.tensor(covs).float(), rcond=1e-10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx], self.ln_covs[idx], self.slow_proj[idx]

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
        sde = cls.system.sde

        # seed setting
        rng = np.random.default_rng(cls.seed)
        rng.integers(10**3);  # warm up of RNG

        # solver
        em = spaths.EulerMaruyama(rng)

        sol =  em.solve(sde, np.array([cls.x0]), cls.tspan, cls.dt)
        path = sol.p[0]

        # skip first few samples and from the rest take only a third
        # t_data = sol.t[1::10]
        data = path[1::10].astype(np.float32)

        data_t = torch.from_numpy(data).float()

        # compute local noise covariances at data points
        covs = cls.system.eval_lnc(data, em, cls.burst_size, cls.burst_dt)
        # TODO: store covariances and use transform parameter to invert while
        #       reading the data
        covi_t = torch.pinverse(torch.tensor(covs).float(), rcond=1e-10)

        # project data points on the slow manifold
        ndt = 10
        nrep = 500
        data_rep = np.repeat(data, nrep, axis=0)
        bursts = em.burst(sde, data_rep, (0, ndt), cls.burst_dt).reshape(len(data), nrep, 2)
        slow_t = torch.from_numpy(np.nanmean(bursts, axis=1)).float()

        data_train, data_test, covi_train, covi_test, slow_train, slow_test = train_test_split(
            data_t, covi_t, slow_t, test_size=0.33, random_state=rng.integers(1e3)
        )

        return (sol.t, path), (data_train, covi_train, slow_train), (data_test, covi_test, slow_test)
