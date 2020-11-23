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

def plot_path_data(axs, times, path, data, clabs=['x', 'y']):
    '''
    Plots path coordinates vs time on axs[0] and data points on axs[1]
    '''
    cl1, cl2 = clabs
    axs[0].plot(times, path.T[0], label=rf"${cl1}$", c=cslow, zorder=2)
    axs[0].plot(times, path.T[1], label=rf"${cl2}$", c=cfast, zorder=1)
    axs[0].set_xlabel("time")
    axs[0].set_title("Evolution of coordinates")
    axs[0].legend()

    axs[1].scatter(*data.T, c=cdata)
    axs[1].set_xlabel(rf"${cl1}$")
    axs[1].set_ylabel(rf"${cl2}$", rotation=0)
    axs[1].set_title("Data points")

dataset = datasets.Sin2(root / 'data')
eps = dataset.eps


hid_sde = Sin2DSystem(eps, hidden=True)

# seed setting and solver
seed = 3579
rng = np.random.default_rng(seed)
rng.integers(10**3);  # warm up of RNG

# stochastic solver
em = spaths.EulerMaruyama(rng)

# simulation params
dt = eps / 4
x0, y0 = 1.0, 1.0
tspan = (0.0, 200.0)

ens0 = np.array([[x0,y0]])
sol_sin2d = em.solve(hid_sde, ens0, tspan, dt)

times, path = sol_sin2d.t, sol_sin2d.p[0]

# periodic boundaries for x
x, y = path.T
x = np.mod(x, 2*np.pi)
path = np.array([x, y]).T

data = path[2::200].astype(np.float32)

fig, axs = plt.subplots(ncols=2, figsize=scale_figsize(width=4/3))
plot_path_data(axs, sol_sin2d.t, path, data, clabs=['y','z'])
axs[0].set_xlim([0, 200])
axs[1].set_xlim([0,2*np.pi])
axs[1].set_ylim([-3.5,3.5])

plt.tight_layout()
plt.savefig(figs_path / 'sin2d_data_hidden.pdf')
plt.close()


## observed raw path and data
times, path = torch.load(data_path / 'raw' / 'path.pt')
data_train, *rest = torch.load(data_path / 'processed' / 'train.pt')
data_test, *rest = torch.load(data_path / 'processed' / 'test.pt')
data = np.vstack([data_train, data_test])

# periodic boundaries for x
x, y = path.T
x = np.mod(x, 2*np.pi)
path = np.array([x, y]).T

fig, axs = plt.subplots(ncols=2, figsize=scale_figsize(width=4/3))
plot_path_data(axs, times, path, data, clabs=['x^1','x^2'])
axs[0].set_xlim([0, 200])
axs[1].set_xlim([0,2*np.pi])
axs[1].set_ylim([-3.5,3.5])
# axs[1].set_xlim([-1.0,1.0])
# axs[1].set_ylim([-1.1,1.5])

plt.tight_layout()
plt.savefig(figs_path / 'sin2d_data_observed.pdf')
plt.close()
