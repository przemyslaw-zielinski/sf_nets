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
from .base import SimDataset
import spaths
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

class RQP4(SimDataset):

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

    # bursts
    burst_size = 10**4
    burst_dt = dt / 2

    # data files
    train_file = 'train.pt'
    test_file = 'test.pt'

    def load(self, data_path):

        data, covs, proj = torch.load(data_path)

        self.data = data
        self.precs = torch.pinverse(covs, rcond=1e-10)
        self.slow_proj = proj

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (point, prec, proj) where
                -> point is data point at 'idx',
                -> prec is the associated inverse of local noise covariance,
                -> proj is the associated projection onto slow manifold
        """
        return self.data[idx], self.precs[idx], self.slow_proj[idx]

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

        sde = cls.system.sde

        # seed setting
        rng = np.random.default_rng(cls.seed)
        rng.integers(10**3);  # warm up of RNG

        # solver
        em = spaths.EulerMaruyama(rng)

        sol =  em.solve(sde, np.array([cls.x0]), cls.tspan, cls.dt)
        times = sol.t
        path = sol.p[0]

        # skip first few samples and from the rest take only a third
        t_data = sol.t[100::3]
        data = np.squeeze(sol(t_data)).astype(dtype=np.float32)
        data_t = torch.from_numpy(data).float()

        # compute local noise covariances at data points
        covs = cls.system.eval_lnc(data, em, cls.burst_size, cls.burst_dt)
        covs_t = torch.from_numpy(covs).float()

        # compute projections of data points on the slow manifold
        nrep = 250
        ndt = int(5*cls.tsep/cls.burst_dt)
        data_rep = np.repeat(data, nrep, axis=0)
        bursts = em.burst(sde, data_rep, (0, ndt), cls.burst_dt)
        bursts = bursts.reshape(len(data), nrep, cls.system.ndim)
        slow_t = torch.from_numpy(np.nanmean(bursts, axis=1)).float()

        data_train, data_test, covs_train, covs_test, slow_train, slow_test \
        = train_test_split(
            data_t, covs_t, slow_t, test_size=0.33, random_state=rng.integers(1e3)
        )

        return (
            (times, path),
            (data_train, covs_train, slow_train),
            (data_test, covs_test, slow_test)
            )
