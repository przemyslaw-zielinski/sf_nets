#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 4 Feb 2021

@author: Przemyslaw Zielinski
"""

script_name = "quad4d_recon_encod"

import sys
from pathlib import Path
root = Path.cwd()
sys.path[0] = str(root)
data_path = root / 'data' / 'Quad4'
figs_path = root / 'results' / 'figs' / 'quad4d'
figs_path.mkdir(exist_ok=True)
model_path = root / 'results' / 'models' / 'Quad4'

import torch
import numpy as np
import matplotlib as mpl
import sf_nets.models as models
import sf_nets.datasets as datasets
from matplotlib import pyplot as plt
from utils.mpl_utils import scale_figsize, to_grid, to_darray

# matplotlib settings
plt.style.use("utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors
PI = np.pi

dataset = datasets.Quad4(root / 'data', train=False)
slow_map = dataset.system.slow_map

dat_t = dataset.data
dat_np = dat_t.detach().numpy()

slow_var = slow_map(dat_np.T)

model_ids = ["mse_elu_1", "mse_elu_2", "mse_elu_3"]
titles = [f"Model {n+1}" for n in range(3)]
fig, axs = plt.subplots(ncols=3, nrows=2, sharey="row",
                        figsize=scale_figsize(width=4/3, height=1.5),
                        # subplot_kw=dict(box_aspect=1),
                        gridspec_kw=dict(wspace=0.1)
                        )


leftmost = True
for ax, model_id, title in zip(axs.T, model_ids, titles):
    # get all info from saved dict
    model_data = torch.load(model_path / f'{model_id}.pt')
    model_arch = model_data['info']['architecture']
    model_args = model_data['info']['arguments']
    state_dict = model_data['best']['model_dict']

    # init model
    model = getattr(models, model_arch)(**model_args)
    model.load_state_dict(state_dict)

    rec_np = model(dat_t).detach().numpy()

    ax[0].scatter(*dat_np.T[:2], label="reconstruction", c=cdata, zorder=0)
    ax[0].scatter(*rec_np.T[:2], label="reconstruction", c=cslow, zorder=1)


    ax[0].set_title(title)
    ax[0].set_xlim([-1.1, 6.1])
    ax[0].set_ylim([-2.1, 2.1])
    ax[0].set_xticks([-1, 6])
    ax[0].set_xlabel(r"$x^1$", labelpad=-6)
    if leftmost:
        ax[0].set_ylabel(r"$x^2$", rotation=0)
        ax[0].set_yticks([-2, 2])

    lat_var = model.encoder(dat_t).detach().numpy().T
    ax[1].scatter(slow_var, lat_var, c=cslow)
    ax[1].set_xlabel('slow variable')
    if leftmost:
        ax[1].set_ylabel('slow view', labelpad=0)
        leftmost = False

plt.savefig(figs_path / f"{script_name}_recon.pdf")#, bbox_inches='tight')
plt.close(fig)
