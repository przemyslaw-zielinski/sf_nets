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

import sf_nets.models as models
import sf_nets.datasets as datasets

from sf_nets.utils.io_utils import io_path, get_script_name
from sf_nets.utils.mpl_utils import scale_figsize, to_grid, to_darray

# PI = np.pi
ds_name = 'Cresc2'

io_path = io_path(ds_name)
script_name = get_script_name()

# matplotlib settings
plt.style.use("sf_nets/utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors

dataset = getattr(datasets, ds_name)(io_path.dataroot, train=False)
slow_map = dataset.system.slow_map

dat_t = dataset.data
dat_np = dat_t.detach().numpy()

slow_var = slow_map(dat_np.T)

model_ids = ["mse_elu_0", "mah_elu_0"]
titles = ["MSE Loss", "Mahalanobis Loss"]
fig, axs = plt.subplots(ncols=2, nrows=2, sharey="row",
                        figsize=scale_figsize(height=1.5),
                        subplot_kw=dict(box_aspect=1),
                        gridspec_kw=dict(wspace=0.05)
                        )


leftmost = True
for ax, model_id, title in zip(axs.T, model_ids, titles):
    # get all info from saved dict
    model_data = torch.load(io_path.models / f'{model_id}.pt')
    model_arch = model_data['info']['architecture']
    model_args = model_data['info']['arguments']
    state_dict = model_data['best']['model_dict']

    # init model
    model = getattr(models, model_arch)(**model_args)
    model.load_state_dict(state_dict)

    # enc_arch = "-".join(model.features[:model.features.index(1)+1])
    # enc_arch = [ str(f) for f in model.features[:model.features.index(1)+1] ]

    rec_np = model(dat_t).detach().numpy()
    # slow manifold
    # x = np.linspace(0.0, 2*PI, 200)
    # ax[0].plot(x + np.sin(np.sin(x)), np.sin(x), '--', c=cslow, lw=1,
    #             label="slow manifold")
    # ax[0].scatter(*dat_np.T, label="data point", c=cdata)
    x = np.linspace(0, 2*np.pi, 100)
    ax[0].plot(np.cos(x), np.sin(x), '--', c=cslow, zorder=0)

    ax[0].scatter(*rec_np.T, label="reconstruction", c=cdata, zorder=1)


    ax[0].set_title(title)
    ax[0].set_xlim([-1.1, 1.1])
    ax[0].set_ylim([-1.1, 1.1])
    ax[0].set_xticks([-1, 1])
    ax[0].set_xlabel(r"$x^1$", labelpad=-6)
    if leftmost:
        ax[0].set_ylabel(r"$x^2$", rotation=0)
        ax[0].set_yticks([-1, 1])
        # leftmost = False
    # if not leftmost:
    #     ax[0].legend(loc = (-0.55, 0.7), framealpha=0.95)


    lat_var = model.encoder(dat_t).detach().numpy().T
    ax[1].scatter(slow_var, lat_var, c=cslow)
    ax[1].set_xlabel('slow variable')
    if leftmost:
        ax[1].set_ylabel('slow view', labelpad=0)
        leftmost = False
    # else:
    #     ax.set_yticks([])
# ax.legend()
# plt.tight_layout()
plt.savefig(io_path.figs / f"{script_name}_recon.pdf")#, bbox_inches='tight')
plt.close(fig)

nlevels = 50
fig, axs = plt.subplots(ncols=3, sharey="row",
                        figsize=scale_figsize(width=4/3, height=0.75),
                        subplot_kw=dict(box_aspect=1),
                        gridspec_kw=dict(wspace=0.1)
                        )
mesh_size = 400
x = np.linspace(-1.5, 1.5, mesh_size)
y = np.linspace(-1.5, 1.5, mesh_size)
X, Y = np.meshgrid(x, y)

mesh_data = to_darray(X, Y)
data_area = (X**2 + Y**2 > .5) * (X**2 + Y**2 < 1.5) * (np.arctan2(Y, X) > -0.9)

s = slow_map(mesh_data.T).T
S = np.squeeze(to_grid(s, (mesh_size, mesh_size)))

axs[0].contour(X, Y, S, levels=np.linspace(-5, 5, 50), colors=cfast,
linewidths=.5, linestyles='solid', alpha=.3)

Sdata = np.ma.masked_where(np.logical_not(data_area), S)
axs[0].contour(X, Y, Sdata, levels=np.linspace(-5, 5, 50), colors=cfast,
    linewidths=1, linestyles='solid')

axs[0].set_title("Slow map")
axs[0].set_ylabel(r"$x^2$", rotation=0, labelpad=-5)

mesh_data = torch.from_numpy(mesh_data).float()
for ax, model_id, title in zip(axs[1:], model_ids, titles):
    model_data = torch.load(path.models / f'{model_id}.pt')
    model_arch = model_data['info']['architecture']
    model_args = model_data['info']['arguments']
    state_dict = model_data['best']['model_dict']

    # init model
    model = getattr(models, model_arch)(**model_args)
    model.load_state_dict(state_dict)

    with torch.no_grad():
        v = model.encoder(mesh_data)

    V = torch.squeeze(to_grid(v, (mesh_size, mesh_size)))
    ax.contour(X, Y, V, '-',
        levels=np.linspace(-6, 6, 70),
        colors=cdata,
        linewidths=.5,
        linestyles='solid',
        alpha=.3)

    Vdata = np.ma.masked_where(np.logical_not(data_area), V)
    ax.contour(X, Y, Vdata, '-', levels=np.linspace(-6, 6, 70),
                             colors=cdata, linewidths=1, linestyles='solid')

    # ax.scatter(*dat_np.T)
    ax.set_title(title)

for ax in axs:
    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])
    ax.set_xlabel(r"$x^1$", labelpad=-6)

plt.savefig(path.figs / f"{script_name}_fibers.pdf")#, bbox_inches='tight')
plt.close(fig)
