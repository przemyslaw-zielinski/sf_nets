"""
Created on Thu 10 Dec 2020

@author: Przemyslaw Zielinski
"""

import torch
import numpy as np
import spaths
from .base import SimDataset, slow_proj
from sf_nets.systems.cresc2d import Cresc2DSystem
from sklearn.model_selection import train_test_split

class Cresc2(SimDataset):

    # system parameters
    a1 = 1.0e-3
    a2 = 1.0e-3
    a3 = 2.5e-2
    a4 = 2.5e-2

    # underlying stochastic system
    system = Cresc2DSystem(a1, a2, a3, a4)

    # simulation parameters
    seed = 9753
    dt = 0.1
    x0 = 1.0, 0.0
    tspan = (0.0, 40_000.0)

    # parameters for slow projections
    nreps = 2_500
    burst_tspan = (0, 500*dt)
    burst_dt = dt

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

        t_data = sol.t[2::160]
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
            data_t, covs_t, slow_t, test_size=0.2, random_state=rng.integers(1e3)
        )

        return (
            (times, path),
            (data_train, covs_train, slow_train),
            (data_test, covs_test, slow_test)
            )

    def load(self, data_path):

        data, covs, proj = torch.load(data_path)

        dx, dy = data.T
        sub = np.arctan2(dy, dx)>-1.0
        data = data[sub]
        covs = covs[sub]
        proj = proj[sub]

        precs = torch.pinverse(covs, rcond=1e-10)
        # evals, evecs = torch.symeig(precs, eigenvectors=False)
        # precs = (precs.T / evals[:, -1]).T

        self.data = data
        self.precs = precs
        self.slow_proj = proj

        # self.data = data
        # self.precs = torch.pinverse(covs, rcond=1e-10)
        # self.slow_proj = slow_proj

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
