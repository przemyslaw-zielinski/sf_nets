#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 18 Nov 2020

@author: Przemyslaw Zielinski
"""

import sys, os
sys.path[0] = os.getcwd()

import torch
import numpy as np
import matplotlib as mpl
import sf_nets.models as models
import sf_nets.datasets as datasets
from matplotlib import pyplot as plt
from utils.mpl_utils import scale_figsize
from utils.io_utils import io_path, get_script_name

ds_name ='Sin2'
io_path = io_path(ds_name)
script_name = get_script_name()

# matplotlib settings
plt.style.use("utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors

model_type = 'mse_elu'
n_ver = 2

dataset = getattr(datasets, ds_name)(io_path.dataroot, train=False)  # use test ds
slow_map = dataset.system.slow_map

dat_t = dataset.data
dat_np = dat_t.detach().numpy()

slow_var = slow_map(dat_np.T)
slow_proj = dataset.slow_proj

model_ids = [model_type + f"_{n}" for n in range(n_ver)]

fig, axs = plt.subplots(ncols=n_ver, nrows=2, sharey='row',
                        figsize=scale_figsize(width=4/3, height=1.7),
                        subplot_kw=dict(box_aspect=1),
                        gridspec_kw=dict(wspace=0.1, hspace=0.3))
leftmost = True
for n, (ax, model_id) in enumerate(zip(axs[0], model_ids)):
    # get all info from saved dict
    model_data = torch.load(io_path.models / f'{model_id}.pt')
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
    model_data = torch.load(io_path.models / f'{model_id}.pt')
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
plt.savefig(io_path.figs / f"{script_name}_{model_type}.pdf")#, bbox_inches='tight')
plt.close()
