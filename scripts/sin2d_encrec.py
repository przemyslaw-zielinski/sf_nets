#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 18 Nov 2020

@author: Przemyslaw Zielinski
"""

import sys
from pathlib import Path
root = Path.cwd()
sys.path[0] = str(root)
data_path = root / 'data' / 'Sin2'
figs_path = root / 'results' / 'figs' / 'sin2d'
model_path = root / 'results' / 'models' / 'Sin2'

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

model_type = "mah_tanh"

dataset = datasets.Sin2(root / 'data', train=False)  # use test ds
slow_map = dataset.system.slow_map

dat_t = dataset.data
dat_np = dat_t.detach().numpy()

slow_var = slow_map(dat_np.T)
slow_proj = dataset.slow_proj

model_ids = [model_type + f"_{n}" for n in range(3)]

fig, axs = plt.subplots(ncols=3, nrows=2, sharey='row',
                        figsize=scale_figsize(width=4/3, height=1.7),
                        subplot_kw=dict(box_aspect=1),
                        gridspec_kw=dict(wspace=0.1, hspace=0.3))
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

    # enc_arch = "-".join(model.features[:model.features.index(1)+1])
    enc_arch = [ str(f) for f in model.features[:model.features.index(1)+1] ]

    lat_var = model.encoder(dat_t).detach().numpy().T
    ax.scatter(slow_var, lat_var, c=cslow)

    # ax.set_title(f"Encoder: " + "-".join(enc_arch))
    ax.set_title(f"Model {n+1}")
    ax.set_xlabel('slow variable')
    if leftmost:
        ax.set_ylabel('slow view', labelpad=0)
        leftmost = False
    # else:
    #     ax.set_yticks([])

leftmost = True
for ax, model_id in zip(axs[1], model_ids):
    # get all info from saved dict
    model_data = torch.load(model_path / f'{model_id}.pt')
    model_arch = model_data['info']['architecture']
    model_args = model_data['info']['arguments']
    state_dict = model_data['best']['model_dict']

    # init model
    model = getattr(models, model_arch)(**model_args)
    model.load_state_dict(state_dict)

    # enc_arch = "-".join(model.features[:model.features.index(1)+1])
    enc_arch = [ str(f) for f in model.features[:model.features.index(1)+1] ]

    rec_np = model(dat_t).detach().numpy()
    ax.scatter(*slow_proj.T, label="projection", c=cslow)
    ax.scatter(*rec_np.T, label="reconstruction", c=cdata)

    ax.set_xlabel(r"$x^1$")
    if leftmost:
        ax.set_ylabel(r"$x^2$", rotation=0)
        # ax.legend(loc='upper right', bbox_to_anchor=(1.5, 0.5),zorder=0)
        leftmost = False

# plt.tight_layout()
plt.savefig(figs_path / f"sin2d_{model_type}_encrec.pdf")#, bbox_inches='tight')
plt.close()
