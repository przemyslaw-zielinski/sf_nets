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

from utils import spaths
from utils.mpl_utils import scale_figsize
from utils.io_utils import io_path, get_script_name

ds_name = 'Quad10'
io_path = io_path(ds_name)
script_name = get_script_name()

# matplotlib settings
plt.style.use("utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors

def to_RGB(c):
    c_rgb = mpl.colors.to_rgb(c)

    c_RGB = (int(v*255) for v in c_rgb)
    return tuple(c_RGB)

print(to_RGB(cdata), to_RGB(cslow), to_RGB(cfast))

def remove_mask(model_dict):
    mask_state_dict = dict(filter(
        lambda elem: elem[0].endswith('_mask'), model_dict.items()
        ))
    orig_state_dict = dict(filter(
        lambda elem: elem[0].endswith('_orig'), model_dict.items()
        ))
    rest = dict(filter(
        lambda elem: elem[0].endswith(('weight', 'bias')), model_dict.items()
        ))
    state_dict = {
        key.replace('_orig',''): val_orig * val_mask
        for (key, val_orig), val_mask in zip(orig_state_dict.items(),
                                             mask_state_dict.values())
    }
    return {**state_dict, **rest}


# model_type = "mahl1_elu"
dataset = getattr(datasets, ds_name)(io_path.dataroot, train=False)  # use test ds

# sdim = dataset.system.sdim
# ndim = dataset.system.ndim
# fdim = ndim - sdim
#
# slow_map = dataset.system.slow_map
# slow_map_der = dataset.system.slow_map_der
#
dat_t = dataset.data
dat_np = dat_t.detach().numpy()
#
# test_precs = dataset.precs
# test_evals, test_evecs = torch.symeig(test_precs, eigenvectors=True)
# f_evecs = test_evecs[:, :, :fdim]

# model_ids = [f"mse_elu_{n}" for n in range(3)]
model_ids = ['mse_elu_2_pruned_nib_r2', 'mse_elu_3_pruned_nib', 'mse_elu_4_pruned_nib']
model_labs = ["Model 3p", "Model 4p", "Model 5p"]
# model_id = model_ids[0]

fig, axs = plt.subplots(ncols=len(model_ids),
    figsize=scale_figsize(width=4/3, height=.75),
    gridspec_kw={'wspace': 0.1, 'width_ratios': [8, 8, 12]},
    sharey=True
)

for ax, model_id, model_lab in zip(axs, model_ids, model_labs):
    model_data = torch.load(io_path.models / f'{model_id}.pt')
    model_arch = model_data['info']['architecture']
    model_args = model_data['info']['arguments']
    state_dict = remove_mask(model_data['best']['model_dict'])

    model = getattr(models, model_arch)(**model_args)
    model.load_state_dict(state_dict)
    model.eval()

    clim = (0, 1)
    ndim = dataset.system.ndim

    weights = model.encoder.layer1.weight.detach().numpy().T

    Z = np.zeros(weights.shape + (3,))
    Z[weights != 0] = mpl.colors.to_rgb(cslow)
    Z[weights == 0] = mpl.colors.to_rgb(cdata)


    ax.imshow(Z, clim=clim, aspect="auto")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(weights.shape[1])-.5, minor=True)
    ax.set_yticks(np.arange(weights.shape[0])-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.set_title(model_lab)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", bottom=False, left=True)

    ax.set_xticks([])
    ax.set_yticks([i for i in range(ndim)])
    ax.set_yticklabels([fr"$x_{{{i+1}}}$" for i in range(ndim)])

    for n, tl in enumerate(ax.get_yticklabels()):
        if n < 4:
            tl.set_color(cslow)

# plt.tight_layout()
plt.savefig(io_path.figs / f"{script_name}.pdf")
plt.close()
