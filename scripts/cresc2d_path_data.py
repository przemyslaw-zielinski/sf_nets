#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 4 Feb 2021

@author: Przemyslaw Zielinski
"""

import sys
from pathlib import Path
root = Path.cwd()
sys.path[0] = str(root)
data_path = root / 'data' / 'Cresc2'
figs_path = root / 'results' / 'figs' / 'cresc2d'
figs_path.mkdir(exist_ok=True)
script_name = "cresc2d_path_data"

import torch
import numpy as np
import matplotlib as mpl
import utils.spaths as spaths
import sf_nets.datasets as datasets
from matplotlib import pyplot as plt
from utils.mpl_utils import scale_figsize
from sf_nets.systems.cresc2d import Cresc2DSystem

# matplotlib settings
plt.style.use("utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors

train_ds = datasets.Cresc2(root / 'data')
times, path = torch.load(data_path / 'raw' / 'path.pt')

fig, axs = plt.subplots(ncols=2,
                        figsize=scale_figsize(width=4/3, height=1.0),
                        sharey=True)

axs[0].plot(*path[:100_000].T, c=cdata)
axs[0].set_title("Sample path")
axs[0].set_ylabel(r"$x^2$", rotation=0, labelpad=-5)

axs[1].scatter(*train_ds.data.T, c=cdata, label="data point")
axs[1].scatter(*train_ds.slow_proj.T, c=cslow, label="projection")
axs[1].set_title("Train data")
axs[1].legend()

for ax in axs:
    ax.set_xlabel(r"$x^1$", labelpad=-5)
    ax.set_aspect('equal')
    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])

plt.savefig(figs_path / f"{script_name}.pdf")
plt.close()
