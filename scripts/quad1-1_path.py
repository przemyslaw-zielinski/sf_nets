#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 11 Feb 2021

@author: Przemyslaw Zielinski
"""

import sys, os
sys.path[0] = os.getcwd()

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from sf_nets.systems.quad import QuadSystem

from utils import spaths
from utils.io_utils import io_path, get_script_name
from utils.mpl_utils import scale_figsize, to_grid, to_darray

ds_name = 'Quad1-1'
script_name = get_script_name()
io_path = io_path(ds_name)

# matplotlib settings
plt.style.use("utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors
PI = np.pi

def slow_proj(data, sde, nreps, tspan, dt):
    nsam, ndim = data.shape
    ens0 = np.repeat(data, nreps, axis=0)
    nsteps = int(tspan[1]/dt)
    bursts = em.burst(sde, ens0, (0, nsteps), dt).reshape(nsam, nreps, ndim)
    return np.nanmean(bursts, axis=1)

# system
Ds = 1
Df = 1
eps = 0.001
sde = QuadSystem(Ds, Df, eps)

# simulation parameters
seed = 9735
dt = eps / 30
x0 = [1.0]*Ds + [0.0]*Df
tspan = (0.0, 0.4)

# seed setting
rng = np.random.default_rng(5891)
rng.integers(10**3);  # warm up of RNG

# solver
em = spaths.EulerMaruyama(rng)

sol =  em.solve(sde, np.array([x0]), tspan, dt)
times = sol.t
path = sol.p[0]
data = path[::10]

slow_map = sde.slow_map

fig, axs = plt.subplots(ncols=2,
                        figsize=scale_figsize(width=4/3, height=0.8),
                        sharey=True)

axs[0].plot(*path[:5_000].T, lw=.5)
axs[0].set_xticks([1, 7])
axs[0].set_yticks([-2, 2])

axs[0].set_xlim([0.5, 7.5])
axs[0].set_ylim([-2.5,2.5])

axs[0].set_title("Sample path")
axs[0].set_xlabel(r"$x^1$", labelpad=-5)
axs[0].set_ylabel(r"$x^2$", rotation=0, labelpad=-5)
axs[0].set_aspect('equal')

x = np.linspace(0.5, 7.5, 100)
plot = axs[1].plot(x, np.zeros_like(x), c=cslow)

mesh_size = 400
x = np.linspace(0.5, 7.5, mesh_size)
y = np.linspace(-2.5, 2.5, mesh_size)
X, Y = np.meshgrid(x, y)

mesh_data = to_darray(X, Y)
s = slow_map(mesh_data.T).T
S = np.squeeze(to_grid(s, (mesh_size, mesh_size)))
cntr = axs[1].contour(X, Y, S, levels=np.linspace(-5.15, 7.6, 30), colors=cfast,
                        linewidths=.5, linestyles='solid', alpha=.3)
cntr_handle, _ = cntr.legend_elements()

axs[1].set_xticks([1, 7])
axs[1].set_title("Slow-fast foliation")
axs[1].set_xlabel(r"$x^1$", labelpad=-5)
axs[1].set_aspect('equal')

axs[1].legend(
    [plot[0], cntr_handle[0]], ["slow manifold", "fast fibers"], loc=(0.02, 0.02))

plt.savefig(io_path.figs / f"{script_name}.pdf")
