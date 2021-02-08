#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 4 Feb 2021

@author: Przemyslaw Zielinski
"""

script_name = "quad4d_fibers"

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

def get_meshslice(coord1, coord2, grid_size, ndim):

    idx1, lim1 = coord1
    idx2, lim2 = coord2
    x = [0.0] * ndim
    x[idx1] = np.linspace(*lim1, grid_size)
    x[idx2] = np.linspace(*lim2, grid_size)

    return np.meshgrid(*x)

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
fig, axs = plt.subplots(ncols=3, sharey="row",
                        figsize=scale_figsize(width=4/3, height=0.75),
                        subplot_kw=dict(box_aspect=1),
                        gridspec_kw=dict(wspace=0.1)
                        )

axs[0].set_ylabel(r"$x^2$", rotation=0, labelpad=-5)
# # axs[0].set_yticks([-3, 3])

for ax, model_id, title in zip(axs, model_ids, titles):
    model_data = torch.load(model_path / f'{model_id}.pt')
    model_arch = model_data['info']['architecture']
    model_args = model_data['info']['arguments']
    state_dict = model_data['best']['model_dict']

    # init model
    model = getattr(models, model_arch)(**model_args)
    model.load_state_dict(state_dict)

    with torch.no_grad():
        v = model.encoder(mesh_data)

    V = torch.squeeze(to_grid(v, mesh_shape))
    X = np.squeeze(XX[id1])
    Y = np.squeeze(XX[id2])
    ax.contour(X, Y, V, '-', levels=np.linspace(-6, 6, nlevels),
                             colors=cdata, linewidths=.5)
    ax.set_title(title)
    ax.set_xlim([-1.1, 6.1])
    ax.set_ylim([-2.1, 2.1])
    ax.set_xticks([-1, 6])
    ax.set_yticks([-2, 2])
    ax.set_xlabel(r"$x^1$", labelpad=-6)

plt.savefig(figs_path / f"{script_name}_fibers.pdf")#, bbox_inches='tight')
plt.close(fig)
