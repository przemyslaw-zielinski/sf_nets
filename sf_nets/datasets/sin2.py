"""
Created on Sat 31 Oct 2020

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


class Sin2System():

    nmd = 2
    ndim = 2
    sdim = 1

    @staticmethod
    def slow_map(z):
        x, y = z
        return x - np.sin(y)

    @staticmethod
    def _drift(eps):

        def drift(t, z, dz):
            x, y = z
            dz[0] = np.sin(y) + np.cos(y)*(np.sin(x-np.sin(y))-y)/eps - np.sin(y)/(2*eps)
            dz[1] = (np.sin(x-np.sin(y))-y) / eps

        return drift

    @staticmethod
    def _dispersion(eps):

        def disp(t, z, dz):
            x, y = z
            dz[0,0] = np.sqrt(1 + .5*np.sin(y))
            dz[0,1] = np.cos(y)/np.sqrt(eps)
            dz[1,1] = 1.0 / np.sqrt(eps)

        return disp

    def __init__(self, eps):

        # observed system
        self.sde = spaths.ItoSDE(self._drift(eps),
                                self._dispersion(eps),
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

class Sin2(SimDataset):

    # system parameters
    eps = 0.001

    # system
    system = Sin2System(eps)
    x0 = [3.0, 3.0]

    # for data normalization
    data_means = [np.pi, 0]
    data_sdevs = [1.0, 1.0]

    # simulation parameters
    seed = 3579
    dt = eps / 4
    tspan = (0.0, 200.0)

    # lnc computations
    burst_size = 10**4
    burst_dt = dt / 2

    def load(self, data_path):
        # self.data, self.precs = torch.load(data_path)
        data, covs, slow_proj = torch.load(data_path)

        data_np = data.detach().numpy()
        slow_proj_np = slow_proj.detach().numpy()
        # means = jnp.mean(data_np, axis=0)
        # stds = 3*jnp.std(data_np, axis=0)
        m0, m1 = self.data_means
        s0, s1 = self.data_sdevs

        fwdZ = lambda x: jnp.array([(x[0] - m0) / (3*s0), (x[1] - m1) / (3*s1)])
        bwdZ = lambda y: jnp.array([3*s0 * y[0] + m0, 3*s1 * y[1] + m1])

        Zdata_np = np.array(fwdZ(data_np.T).T)
        Zslow_proj_np = np.array(fwdZ(slow_proj_np.T).T)
        self.data = torch.from_numpy(Zdata_np)
        self.slow_proj = torch.from_numpy(Zslow_proj_np)

        transformZ = spaths.SDETransform(fwdZ, bwdZ)
        self.system = Sin2System(self.eps)
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
        self.precs = torch.pinverse(torch.tensor(covs).float(), rcond=1e-10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx], self.precs[idx], self.slow_proj[idx]

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

        # folding: x is wrapped into [0, 2pi]
        x, y = path.T
        x = np.mod(x, 2*np.pi)
        f_path = np.array([x, y]).T

        # skip first few samples and from the rest take only a third
        # t_data = sol.t[1::10]
        data = f_path[1::200].astype(np.float32)

        data_t = torch.from_numpy(data).float()

        # compute local noise covariances at data points
        covs = cls.system.eval_lnc(data, em, cls.burst_size, cls.burst_dt)
        # TODO: store covariances and use transform parameter to invert while
        #       reading the data
        # covi_t = torch.pinverse(torch.tensor(covs).float(), rcond=1e-10)
        covs_t = torch.from_numpy(covs).float()

        # compute projections of data points on the slow manifold
        nrep = 250
        ndt = int(3*cls.eps/cls.burst_dt)
        data_rep = np.repeat(data, nrep, axis=0)
        bursts = em.burst(sde, data_rep, (0, ndt), cls.burst_dt)
        bursts = bursts.reshape(len(data), nrep, 2)
        slow_t = torch.from_numpy(np.nanmean(bursts, axis=1)).float()

        data_train, data_test, covs_train, covs_test, slow_train, slow_test = train_test_split(
            data_t, covs_t, slow_t, test_size=0.33, random_state=rng.integers(1e3)
        )

        return (
            (sol.t, path),
            (data_train, covs_train, slow_train),
            (data_test, covs_test, slow_test)
            )
