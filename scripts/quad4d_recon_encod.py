#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 4 Feb 2021

@author: Przemyslaw Zielinski
"""

import sys, os
sys.path[0] = os.getcwd()

import yaml
import torch
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import sf_nets.models as models
import sf_nets.datasets as datasets

from sf_nets.utils.io_utils import io_path, get_script_name
from sf_nets.utils.mpl_utils import scale_figsize, to_grid, to_darray

ds_name = 'Quad4'
script_name = get_script_name()
io_path = io_path(ds_name)

# matplotlib settings
plt.style.use("sf_nets/utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors
PI = np.pi

def get_meshslice(coord1, coord2, grid_size, ndim):

    idx1, lim1 = coord1
    idx2, lim2 = coord2
    x = [0.0] * ndim
    x[idx1] = np.linspace(*lim1, grid_size)
    x[idx2] = np.linspace(*lim2, grid_size)

    return np.meshgrid(*x)

dataset = getattr(datasets, ds_name)(io_path.dataroot, train=False)
slow_map = dataset.system.slow_map

dat_t = dataset.data
dat_np = dat_t.detach().numpy()

slow_var = slow_map(dat_np.T)

mesh_size = 120
x0 = np.linspace(+0.0, 4.0, mesh_size)
x1 = np.linspace(-2.0, 2.0, mesh_size)
x2, x3 = np.zeros((2, 1))
X = np.meshgrid(x0, x1, x2, x3)

coords = [
    (0, [-1.0, 6.0]),
    (1, [-2.0, 2.0]),
    (2, [-2.0, 2.0]),
    (3, [-2.0, 2.0])
]

id1 = 0
id2 = 1
XX = get_meshslice(coords[id1], coords[id2], mesh_size, 4)

mesh_data = torch.tensor(to_darray(*XX), dtype=torch.float)
mesh_shape = XX[0].shape
nlevels = 50

with open("experiments.yaml", 'rt') as yaml_file:
    models_data = yaml.full_load(yaml_file)[ds_name]
model_ids = models_data['model_ids']
model_tags = models_data['model_tags']

fig, axs = plt.subplots(ncols=len(model_ids), nrows=2, sharey="row",
                        figsize=scale_figsize(width=4/3, height=1.5),
                        # subplot_kw=dict(box_aspect=1),
                        gridspec_kw=dict(wspace=0.1)
                        )

leftmost = True
for ax, id, tag in zip(axs.T, model_ids, model_tags):
    # get all info from saved dict
    model_data = torch.load(io_path.models / f'{id}.pt')
    model_arch = model_data['info']['architecture']
    model_args = model_data['info']['arguments']
    state_dict = model_data['best']['model_dict']

    # init model
    model = getattr(models, model_arch)(**model_args)
    model.load_state_dict(state_dict)

    rec_np = model(dat_t).detach().numpy()

    with torch.no_grad():
        v = model.encoder(mesh_data)

    V = torch.squeeze(to_grid(v, mesh_shape))
    X = np.squeeze(XX[id1])
    Y = np.squeeze(XX[id2])
    ax[0].contour(X, Y, V, '-', levels=np.linspace(-6, 6, nlevels),
                             colors=cdata, linewidths=.5,
                             linestyles='solid', zorder=0)

    # ax[0].scatter(*dat_np.T[:2], label="reconstruction", c=cdata, zorder=0)
    ax[0].scatter(*rec_np.T[:2], label="reconstruction", c=cslow, zorder=1)


    ax[0].set_title(tag)
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

plt.savefig(io_path.figs / f"{script_name}_recon.pdf")#, bbox_inches='tight')
plt.close(fig)
