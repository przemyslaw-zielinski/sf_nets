#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 15 Dec 2020

@author: Przemyslaw Zielinski
"""

import sys
from pathlib import Path
root = Path.cwd()
sys.path[0] = str(root)

import torch
import numpy as np
import matplotlib as mpl
import sf_nets.models as models
import sf_nets.datasets as datasets
from matplotlib import pyplot as plt
from utils.mpl_utils import scale_figsize

# matplotlib settings
plt.style.use("utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors

name_ds = 'Quad4'
model_type = "mahl1_elu"

data_path = root / 'data' / name_ds
figs_path = root / 'results' / 'figs' / name_ds.lower()
model_path = root / 'results' / 'models' / name_ds

dataset = getattr(datasets, name_ds)(root / 'data', train=False)  # use test ds
slow_map = dataset.system.slow_map
figs_path.mkdir(exist_ok=True)

dat_t = dataset.data
dat_np = dat_t.detach().numpy()

slow_var = slow_map(dat_np.T)
slow_proj = dataset.slow_proj

model_ids = [model_type + f"_{n}" for n in range(3)]
met_names = ['proj_mse_loss', 'proj_max_loss']
met_labs = ['MSE', 'Max']
max_epochs = 2000

def moving_avg(t, n):
    x = t.detach().numpy()
    cumsum = np.cumsum(np.insert(x, 0, [0]*n))
    return torch.from_numpy((cumsum[n:] - cumsum[:-n]) / float(n))

fig, axs = plt.subplots(nrows=len(met_names), sharex=True,
                        figsize=scale_figsize(width=4/3, height=1.4),
                        gridspec_kw=dict(wspace=0.1, hspace=0.1))
n_smooth = 10
for ax, met_name, met_lab in zip(axs, met_names, met_labs):
    for n, model_id in enumerate(model_ids):

        epochs = range(1, max_epochs+1)

        model_data = torch.load(model_path / f'{model_id}.pt')
        metrics = [
            met[met_name]
            for met in model_data['history']['metrics']
        ]
        metrics = torch.tensor(metrics)
        metrics = moving_avg(metrics, n_smooth)

        ax.plot(epochs, metrics[:max_epochs], c=f'C{n}', label=f'Model {n+1}a')
        ax.set_xlim([1, max_epochs])
        if met_name == 'proj_mse_loss':
            ax.set_yscale('log')

        model_data = torch.load(model_path / f'{model_id}a.pt')

        metrics = [
            met[met_name]
            for met in model_data['history']['metrics']
        ]
        metrics = torch.tensor(metrics)
        metrics = moving_avg(metrics, n_smooth)

        ax.plot(epochs, metrics[:max_epochs],'--', c=f'C{n}', label=f'Model {n+1}b')
        ax.set_xlim([1, max_epochs])


    if met_name == 'proj_max_loss':
        ax.legend(loc='upper right', bbox_to_anchor=(1.04, 1.4), framealpha=0.95)

    ax.set_ylabel(met_lab, rotation=0)

# plt.tight_layout()
fig.align_ylabels(axs)
plt.savefig(figs_path / f"{name_ds.lower()}_{model_type}_metrics.pdf")
plt.close()
