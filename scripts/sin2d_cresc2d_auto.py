#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 8 Feb 2021

@author: Przemyslaw Zielinski
"""

import sys, os
sys.path[0] = os.getcwd()

import torch
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from sf_nets.utils.mpl_utils import scale_figsize, to_grid, to_darray

import sf_nets.models as models
import sf_nets.datasets as datasets

from sf_nets.utils.io_utils import io_path, get_script_name
from sf_nets.utils.mpl_utils import scale_figsize

path = io_path()
script_name = get_script_name()

# matplotlib settings
plt.style.use("sf_nets/utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors
PI = np.pi

sin2_ds = datasets.Sin2(path.dataroot, train=False)
cresc2_ds = datasets.Cresc2(path.dataroot, train=False)

datasets = [sin2_ds, cresc2_ds]
model_ids = ['projmse_elu_0', 'projmse_elu_0']

fig, axs = plt.subplots(ncols=2,
                        figsize=scale_figsize(width=4/3, height=1.1),
                        subplot_kw=dict(box_aspect=1.0),
                        gridspec_kw=dict(wspace=0.55)
                        )


leftmost = True
for ax, dataset, model_id in zip(axs, datasets, model_ids):

    data = dataset.data

    slow_map = dataset.system.slow_map
    slow_var = slow_map(data.detach().numpy().T)

    model_data = torch.load(path.models / dataset.name / f'{model_id}.pt')
    model_arch = model_data['info']['architecture']
    model_args = model_data['info']['arguments']
    state_dict = model_data['best']['model_dict']

    # init model
    model = getattr(models, model_arch)(**model_args)
    model.load_state_dict(state_dict)

    with torch.no_grad():
        pred = model(data)
    sctr = ax.scatter(*pred.T, label="slow manifold", c=cslow, zorder=3)

    axins = ax.inset_axes([0.7, 0.6, 0.6, 0.6])
    axins.patch.set_alpha(0.95)
    axins.tick_params(direction="in", pad=-12)
    axins.set_xticks([])
    axins.set_yticks([])

    if dataset.name == 'Sin2':
        ax.set_xlim([0,2*PI])
        ax.set_ylim([-PI, PI])
        ax.set_xticks([0, 2*PI])
        ax.set_xticklabels(['0', r'$2\pi$'])
        ax.set_yticks([-3, 3])

        xlim = (0.0, 2*PI)
        ylim = (-PI, PI)

        axins.set_xlim([-1.5, 7])
        axins.set_ylim([-8, 3])
        # axins.set_xticks([0, 6])
        # axins.set_yticks([-6, 2])
        # axins.set_xticklabels([0,5])

    if dataset.name == 'Cresc2':
        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])
        ax.set_xticks([-1, 1])
        ax.set_yticks([-1, 1])

        axins.set_xlim([-2.5, 4.0])
        axins.set_ylim([-3.5, 3.0])
        # axins.set_xticks([-1, 4])
        # axins.set_yticks([-2, 2])

        xlim = (-1.3, 1.3)
        ylim = (-1.3, 1.3)

    mesh_size = 400
    x = np.linspace(*xlim, mesh_size)
    y = np.linspace(*ylim, mesh_size)
    X, Y = np.meshgrid(x, y)

    mesh_data = to_darray(X, Y)
    mesh_data = torch.from_numpy(mesh_data).float()
    with torch.no_grad():
        v = model.encoder(mesh_data)
    V = torch.squeeze(to_grid(v, (mesh_size, mesh_size)))

    cntr = ax.contour(X, Y, V, levels=50, colors=cfast, linewidths=.5,
        linestyles='solid')

    cntr_handle, _ = cntr.legend_elements()

    lat_var = model.encoder(data).detach().numpy().T
    insctr = axins.scatter(slow_var, lat_var, c=cdata)

    ax.set_xlabel(r"$x^1$", labelpad=-5)
    ax.set_ylabel(f"$x^2$", labelpad=-4, rotation=0)

    axins.set_xlabel("slow variable", labelpad=-9, fontsize="small")
    axins.set_ylabel("slow view", labelpad=-2, fontsize="small", rotation=-90)

    if dataset.name == 'Sin2':
        ax.legend(
            [sctr, insctr, cntr_handle[0]],
            ["slow manifold reconstruction",
             "encoder vs slow variable",
             "encoder level sets"],
            loc='lower left',
            bbox_to_anchor= (-0.08, -0.3),
            ncol=3,
            frameon=False,
            handlelength=1.0,
            handletextpad=0.5,
            markerscale=2.0,
            markerfirst=False
            # framealpha=0.95
        )

plt.savefig(path.figs / f"{script_name}.pdf")#, bbox_inches='tight')
plt.close(fig)
