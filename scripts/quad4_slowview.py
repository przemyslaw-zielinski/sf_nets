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
from sf_nets.utils.mpl_utils import scale_figsize

# matplotlib settings
plt.style.use("sf_nets/utils/manuscript.mplstyle")
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

fig, axs = plt.subplots(ncols=3, nrows=2, sharey='row', sharex=True,
                        figsize=scale_figsize(width=4/3, height=1.4),
                        subplot_kw=dict(box_aspect=1),
                        gridspec_kw=dict(wspace=0.1, hspace=0.1))
leftmost = True
for n, (ax, model_id) in enumerate(zip(axs[0], model_ids)):
    # get all info from saved dict
    model_data = torch.load(model_path / f'{model_id}.pt')
    model_arch = model_data['info']['architecture']
    model_args = model_data['info']['arguments']
    state_dict = model_data['best']['model_dict']

    # init model
    model = getattr(models, model_arch)(**model_args)
    model.load_state_dict(state_dict)
    lam = model.proj_loss_wght

    lat_var = model.encoder(dat_t).detach().numpy().T
    ax.scatter(slow_var, lat_var, c=cslow)

    ax.set_title(f"Model {n+1}")
    ax.text(0.05, 0.9, rf'{n+1}a ($\lambda=$ {lam})',
            verticalalignment='center', horizontalalignment='left',
            transform=ax.transAxes)
    if leftmost:
        ax.set_ylabel('slow view', labelpad=0)
        leftmost = False
    # else:
    #     ax.set_yticks([])

leftmost = True
for n, (ax, model_id) in enumerate(zip(axs[1], model_ids)):
    # get all info from saved dict
    model_data = torch.load(model_path / f'{model_id}a.pt')
    model_arch = model_data['info']['architecture']
    model_args = model_data['info']['arguments']
    state_dict = model_data['best']['model_dict']

    # init model
    model = getattr(models, model_arch)(**model_args)
    model.load_state_dict(state_dict)
    lam = model.proj_loss_wght

    lat_var = model.encoder(dat_t).detach().numpy().T
    ax.scatter(slow_var, lat_var, c=cslow)

    ax.set_xlabel('slow variable')
    ax.text(0.05, 0.9, rf'{n+1}b ($\lambda=$ {lam})',
            verticalalignment='center', horizontalalignment='left',
            transform=ax.transAxes)
    if leftmost:
        ax.set_ylabel('slow view', labelpad=0)
        leftmost = False

# plt.tight_layout()
plt.savefig(figs_path / f"{name_ds.lower()}_{model_type}_slowview.pdf")#, bbox_inches='tight')
plt.close()
