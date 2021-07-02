#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 17 Nov 2020

@author: Przemyslaw Zielinski
"""

import sys, os
sys.path[0] = os.getcwd()

import torch
import numpy as np
import matplotlib as mpl
import spaths
import sf_nets.datasets as datasets
from matplotlib import pyplot as plt
from sf_nets.utils.mpl_utils import scale_figsize
from sf_nets.systems.sin2d import Sin2DSystem
from sf_nets.utils.io_utils import io_path, get_script_name

ds_name = 'Sin2'
script_name = get_script_name()
io_path = io_path(ds_name)

# matplotlib settings
plt.style.use("sf_nets/utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors
PI = np.pi

def plot_fibs(data, sde, solver, tspan, dt, c=None):
    fib_paths = solver.solve(sde, data, tspan, dt).p
    for fib_path in fib_paths:
        plt.scatter(*fib_path.T, s=7, c=c);

def plot_sman(data, sde, solver, nreps, tspan, dt, c=None):
    ens0 = np.repeat(data, nreps, axis=0)
    nsteps = int(tspan[1]/dt)
    bursts = solver.burst(sde, ens0, (0, nsteps), dt)
    bursts = bursts.reshape(len(data), nreps, 2)
    slow_means = np.nanmean(bursts, axis=1)
    plt.scatter(*slow_means.T, c=c, s=10)

    return slow_means

data_train, *rest = torch.load(io_path.data / 'processed' / 'train.pt')
data_test, *rest = torch.load(io_path.data / 'processed' / 'test.pt')
data = np.vstack([data_train, data_test])
sub_data = data[::20]

eps = getattr(datasets, ds_name).eps
sde = Sin2DSystem(eps)

# seed setting and solver
seed = 3579
rng = np.random.default_rng(seed)
rng.integers(10**3);  # warm up of RNG

# stochastic solver
em = spaths.EulerMaruyama(rng)

# simulation params
dt = eps / 8
tspan = (0.0, 3*eps)

fig, axs = plt.subplots(ncols=2, figsize=scale_figsize(width=4/3))

# slow manifold
nreps = 250
ens0 = np.repeat(sub_data, nreps, axis=0)
nsteps = int(tspan[1]/dt)
bursts = em.burst(sde, ens0, (0, nsteps), dt)
bursts = bursts.reshape(len(sub_data), nreps, 2)
slow_means = np.nanmean(bursts, axis=1)

axs[0].scatter(*slow_means.T, c=cslow)
axs[0].set_xlim([0, 2*PI])
axs[0].set_ylim([-PI, PI])
axs[0].set_xticks([0, PI, 2*PI])
axs[0].set_xticklabels(['0', r'$\pi$', r'$2\pi$'])

axs[0].set_title("Slow manifold")
axs[0].set_xlabel(r"$x^1$")
axs[0].set_ylabel(r"$x^2$", rotation=0)

axs[0].set_aspect('equal')

# fibers
fib_paths = em.solve(sde, sub_data, (0.0, eps), dt/4).p
for fib_path in fib_paths:
    axs[1].scatter(*fib_path.T, c=cfast)
axs[1].set_xlim([0,2*PI])
axs[1].set_ylim([-PI, PI])
axs[1].set_xticks([0, PI, 2*PI])
axs[1].set_xticklabels(['0', r'$\pi$', r'$2\pi$'])

axs[1].set_title("Fast fibers")
axs[1].set_xlabel(r"$x^1$")
axs[1].set_ylabel(r"$x^2$", rotation=0)

axs[1].set_aspect('equal')

plt.tight_layout()
plt.savefig(io_path.figs / f'{script_name}.pdf')
plt.close()
