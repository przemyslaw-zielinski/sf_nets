#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 16 Mar 2021

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

PI = np.pi
ds_name ='Ohm2'
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

model_id = "mse_elu_4"
fig, axs = plt.subplots(ncols=3,
                        figsize=scale_figsize(width=4/3, height=.8),
                        subplot_kw=dict(box_aspect=1),
                        # gridspec_kw=dict(wspace=0.05, hspace=0.35)
                        )

# get all info from saved dict
model_data = torch.load(io_path.models / f'{model_id}.pt')
model_arch = model_data['info']['architecture']
model_args = model_data['info']['arguments']
state_dict = model_data['best']['model_dict']

# init model
model = getattr(models, model_arch)(**model_args)
model.load_state_dict(state_dict)

levels = 50
mesh_size = 400
x = np.linspace(-1.5, 1.5, mesh_size)
y = np.linspace(-1.5, 1.5, mesh_size)
X, Y = np.meshgrid(x, y)

mesh_data = to_darray(X, Y)
accurate_area = (X**2 + Y**2 > .5) * (X**2 + Y**2 < 1.5) * \
                ((np.arctan2(Y, X) > -1.25) + (np.arctan2(Y,X)<-2.4))

# slow manifold
x = np.linspace(0, 2*PI, 100)
axs[0].plot(np.cos(x), np.sin(x), '--', c=cslow, zorder=0, label="slow manifold")
# axs[0].plot(x + np.sin(np.sin(x)), np.sin(x), '--', c=cslow, lw=1,
#             label="slow manifold",zorder=0)

# fast fibers
s = slow_map(mesh_data.T).T
S = np.squeeze(to_grid(s, (mesh_size, mesh_size)))
axs[0].contour(X, Y, S, levels=np.linspace(-5, 5, 50), colors=cfast,
linewidths=.5, linestyles='solid', alpha=.3, zorder=0)

Sdata = np.ma.masked_where(np.logical_not(accurate_area), S)
axs[0].contour(X, Y, Sdata, levels=np.linspace(-5, 5, 50), colors=cfast,
    linewidths=1, linestyles='solid', alpha=.8, zorder=0)

# remove the lines on discontinuity
axs[0].axhline(0,0,0.5, c='w', linestyle='solid', lw=1.5, zorder=0,
    solid_capstyle='round')

rec_np = model(dat_t).detach().numpy()
axs[0].scatter(*rec_np.T, label="reconstruction", c=cdata)

axs[0].set_title("Manifold reconstruction")
axs[0].set_xlim([-1.5, 1.5])
axs[0].set_ylim([-1.5, 1.5])
axs[0].set_xticks([-1, 1])
axs[0].set_yticks([-1, 1])
axs[0].set_xlabel(r"$x^1$", labelpad=-6)
axs[0].set_ylabel(r"$x^2$", rotation=0)
axs[0].set_xlabel(r"$x^1$", labelpad=-4)
axs[0].set_ylabel(r"$x^2$", rotation=0, labelpad=-3)
axs[0].legend(loc="lower left", fontsize='xx-small')

# encoder level sets
mesh_data = torch.from_numpy(mesh_data).float()
with torch.no_grad():
    v = model.encoder(mesh_data)

V = torch.squeeze(to_grid(v, (mesh_size, mesh_size)))
levels = np.linspace(-1.5, 1.5, 70)
axs[1].contour(X, Y, V, '-',
    levels=levels,
    colors=cdata,
    linewidths=.5,
    linestyles='solid',
    alpha=.3)

Vdata = np.ma.masked_where(np.logical_not(accurate_area), V)
axs[1].contour(X, Y, Vdata, '-', levels=levels,
                         colors=cdata, linewidths=1, linestyles='solid')

axs[1].set_title("Encoder: level sets")
axs[1].set_xlabel(r"$x^1$", labelpad=-4)
axs[1].set_ylabel(r"$x^2$", rotation=0, labelpad=-6)
axs[1].set_xlim([-1.5, 1.5])
axs[1].set_ylim([-1.5, 1.5])
axs[1].set_xticks([-1, 1])
axs[1].set_yticks([-1, 1])
# axs[1].set_xticklabels(['0', r'$2\pi$'])
# axs[1].set_yticklabels([])

# encoder latent view
lat_var = model.encoder(dat_t).detach().numpy().T
axs[2].scatter(slow_var, lat_var, c=cdata)

axs[2].set_title("Encoder: slow view")
axs[2].set_xlabel('slow variable', labelpad=-4)
axs[2].set_ylabel('slow view', labelpad=-10)
axs[2].set_yticks([-3, 5])
axs[2].set_xticks([-3, 3])

plt.tight_layout()
plt.savefig(io_path.figs / f"{script_name}.pdf")#, bbox_inches='tight')
plt.close(fig)
