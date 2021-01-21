"""
Created on Thu 10 Dec 2020

@author: Przemyslaw Zielinski
"""

import torch
import numpy as np
import utils.spaths as spaths
from .base import SimDataset, slow_proj
from sf_nets.systems.parab import ParabSystem
from sklearn.model_selection import train_test_split

class Parab3(SimDataset):

    # system parameters
    lam = 1.0
    eta = 1.0
    gam = 1.0
    sig = 0.5
    eps = 0.01
    # lam = 1e-3
    # eta = 1e+0
    # gam = 1e-3
    # sig = 10.0
    # eps = 1e-2

    # underlying stochastic system
    system = ParabSystem(lam, eta, gam, sig, eps, hidden=True)

    # simulation parameters
    seed = 3579
    dt = .3 * eps
    x0 = 0.0, 0.0, 0.0
    tspan = (0.0, 200.0)

    # parameters for slow projections
    nreps = 2_000
    burst_tspan = (0, 2*eps)
    burst_dt = dt / 4

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

        sde = cls.system

        # seed setting
        rng = np.random.default_rng(cls.seed)
        rng.integers(10**3);  # warm up of RNG

        # solver
        em = spaths.EulerMaruyama(rng)

        sol =  em.solve(sde, np.array([cls.x0]), cls.tspan, cls.dt)
        times = sol.t
        path = sol.p[0]

        t_data = sol.t[::15]
        data = np.squeeze(sol(t_data)).astype(dtype=np.float32)
        data_t = torch.from_numpy(data).float()

        # compute local noise covariances at data points
        covs = cls.system.eval_lnc(data)
        covs_t = torch.from_numpy(covs).float()

        # compute projections of data points on the slow manifold
        slow_t = torch.from_numpy(
            slow_proj(data, sde, em, cls.nreps, cls.burst_tspan, cls.burst_dt)
        ).float()

        data_train, data_test, covs_train, covs_test, slow_train, slow_test \
        = train_test_split(
            data_t, covs_t, slow_t, test_size=0.33, random_state=rng.integers(1e3)
        )

        return (
            (times, path),
            (data_train, covs_train, slow_train),
            (data_test, covs_test, slow_test)
            )

    def load(self, data_path):

        data, covs, slow_proj = torch.load(data_path)

        self.data = data
        self.precs = torch.pinverse(covs, rcond=1e-10)
        self.slow_proj = slow_proj

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
