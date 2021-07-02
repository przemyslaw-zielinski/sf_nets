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
from matplotlib import pyplot as plt

import spaths
import sf_nets.datasets as datasets
from sf_nets.utils.mpl_utils import scale_figsize
from sf_nets.systems.sin2d import Sin2DSystem
from sf_nets.utils.io_utils import io_path, get_script_name

ds_name = 'Sin2'
script_name = get_script_name()
io_path = io_path(ds_name)

# matplotlib settings
plt.style.use("sf_nets/utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors
pi = np.pi

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

def to_darray(*meshgrids):
    return np.stack(meshgrids).reshape(len(meshgrids), -1).T

def to_grid(darray, grid_size):
    if darray.ndim == 1:
        return darray.reshape(grid_size, -1)
    else:
        return darray.reshape(darray.shape[1], grid_size, -1)

dataset = getattr(datasets, ds_name)(io_path.dataroot)
slow_map = dataset.system.slow_map
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
x = np.mod(x, 2*pi)
path = np.array([x, y]).T

data = path[2::200].astype(np.float32)

fig, axs = plt.subplots(ncols=2, figsize=scale_figsize(width=4/3))
plot_path_data(axs, sol_sin2d.t, path, data, clabs=['y','z'])
y = np.linspace(0, 2*pi, 100)
z = np.sin(y)
std = np.sqrt(.5)
axs[1].plot(y, z, c=cslow)
axs[1].vlines(y[::9], z[::9] + 3*std, z[::9] - 3*std, colors=cfast, lw=.5)

axs[0].set_xlim([0, 200])
axs[1].set_xlim([0, 2*pi])
axs[1].set_ylim([-pi, pi])
axs[1].set_xticks([0, pi, 2*pi])
axs[1].set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
axs[1].set_aspect('equal')

plt.tight_layout()
plt.savefig(io_path.figs / f'{script_name}_hidden.pdf')
plt.close()


## observed raw path and data
times, path = torch.load(io_path.data / 'raw' / 'path.pt')
data_train, *rest = torch.load(io_path.data / 'processed' / 'train.pt')
data_test, *rest = torch.load(io_path.data / 'processed' / 'test.pt')
data = np.vstack([data_train, data_test])

# periodic boundaries for x
x, y = path.T
x = np.mod(x, 2*pi)
path = np.array([x, y]).T

fig, axs = plt.subplots(ncols=2, figsize=scale_figsize(width=4/3))
plot_path_data(axs, times, path, data, clabs=['x^1','x^2'])
axs[0].set_xlim([0, 200])

# axs[1].set_xlim([-1.0,1.0])
# axs[1].set_ylim([-1.1,1.5])

# fast fbers
mesh_size = 400
x = np.linspace(+0.0, 2*pi, mesh_size)
y = np.linspace(-pi, pi, mesh_size)
X, Y = np.meshgrid(x, y)

mesh_data = to_darray(X, Y)
v = slow_map(mesh_data.T).T
V = np.squeeze(to_grid(v, mesh_size))

axs[1].contour(X, Y, V, levels=10, colors=cfast, linewidths=.5)

# slow manifold
axs[1].plot(x + np.sin(np.sin(x)), np.sin(x), c=cslow, lw=1)

axs[1].set_xlim([0, 2*pi])
axs[1].set_ylim([-pi, pi])
axs[1].set_xticks([0, pi, 2*pi])
axs[1].set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
axs[1].set_aspect('equal')

plt.tight_layout()
plt.savefig(io_path.figs / f'{script_name}_observed.pdf')
plt.close()
