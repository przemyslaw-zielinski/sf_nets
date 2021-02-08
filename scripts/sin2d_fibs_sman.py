#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 17 Nov 2020

@author: Przemyslaw Zielinski
"""

import sys
from pathlib import Path
root = Path.cwd()
sys.path[0] = str(root)
data_path = root / 'data' / 'Sin2'
figs_path = root / 'results' / 'figs' / 'sin2d'

import torch
import numpy as np
import matplotlib as mpl
import utils.spaths as spaths
import sf_nets.datasets as datasets
from matplotlib import pyplot as plt
from utils.mpl_utils import scale_figsize
from sf_nets.systems.sin2d import Sin2DSystem

# matplotlib settings
plt.style.use("utils/manuscript.mplstyle")
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

data_train, *rest = torch.load(data_path / 'processed' / 'train.pt')
data_test, *rest = torch.load(data_path / 'processed' / 'test.pt')
data = np.vstack([data_train, data_test])
sub_data = data[::20]

eps = datasets.Sin2.eps
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
plt.savefig(figs_path / 'sin2d_fibs_sman.pdf')
plt.close()
