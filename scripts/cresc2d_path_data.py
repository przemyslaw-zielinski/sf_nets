#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 4 Feb 2021

@author: Przemyslaw Zielinski
"""

import sys, os
sys.path[0] = os.getcwd()

import torch
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import sf_nets.datasets as datasets
from sf_nets.systems.cresc2d import Cresc2DSystem

import spaths
from sf_nets.utils.mpl_utils import scale_figsize
from sf_nets.utils.io_utils import io_path, get_script_name

# settings
ds_name = 'Cresc2'
iop = io_path(ds_name)
script_name = get_script_name()

# matplotlib settings
plt.style.use("sf_nets/utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors

dataset = getattr(datasets, ds_name)(iop.dataroot)
times, path = torch.load(dataset.raw / 'path.pt')

fig, axs = plt.subplots(ncols=2,
                        figsize=scale_figsize(width=4/3, height=1.0),
                        sharey=True)

axs[0].plot(*path[:100_000].T, c=cdata)
axs[0].set_title("Sample path")
axs[0].set_ylabel(r"$x^2$", rotation=0, labelpad=-5)

axs[1].scatter(*dataset.data.T, c=cdata, label="data point")
axs[1].scatter(*dataset.slow_proj.T, c=cslow, label="projection")
axs[1].set_title("Train data")

# settings
ds_name = 'Ohm2'
iop = io_path(ds_name)
dataset = getattr(datasets, ds_name)(iop.dataroot)
axs[1].scatter(*dataset.data.T, c=cdata, alpha=0.3, zorder=0)
axs[1].scatter(*dataset.slow_proj.T, c=cslow, alpha=0.3, zorder=0)

axs[1].legend(loc=3)

for ax in axs:
    ax.set_xlabel(r"$x^1$", labelpad=-5)
    ax.set_aspect('equal')
    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])

plt.savefig(iop.figs / f"{script_name}.pdf")
plt.close()