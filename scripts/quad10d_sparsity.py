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
model_ids = ['mse_elu_2_pruned']
model_id = model_ids[0]
### Derivatives ###

model_data = torch.load(io_path.models / f'{model_id}.pt')
model_arch = model_data['info']['architecture']
model_args = model_data['info']['arguments']
state_dict = remove_mask(model_data['best']['model_dict'])

model = getattr(models, model_arch)(**model_args)
model.load_state_dict(state_dict)
model.eval()

# dat_t.requires_grad_(True);
# g = torch.eye(sdim).repeat(len(dat_t), 1, 1).T
fig, axs = plt.subplots(nrows=1,#2*len(model_ids),
                        # figsize=scale_figsize(width=.5),
                        subplot_kw={},
                        gridspec_kw={'width_ratios': [.1]})
par_dict = {}
for coder_name, coder in model.named_children():
    for name, par in coder.named_parameters():
        par_np = par.detach().numpy()
        if par_np.ndim == 1:
            par_np = par_np[:, np.newaxis]
        layer_name, par_name = name.split('.')
        value = par_dict.setdefault(coder_name+layer_name, [])

        value.append((par_name, par_np.T))
clim = (0, 1)

ndim = dataset.system.ndim
# plt.setp(axs, xticks=[], yticks=[])
for n, (layer_name, par_list) in enumerate(par_dict.items()):
    if n == 0:
        weights = par_list[0][1]
        axs.imshow(np.absolute(weights) > 0, clim=clim)

        # Turn spines off and create white grid.
        for edge, spine in axs.spines.items():
            spine.set_visible(False)

        axs.set_xticks(np.arange(weights.shape[1])-.5, minor=True)
        axs.set_yticks(np.arange(weights.shape[0])-.5, minor=True)
        axs.grid(which="minor", color="w", linestyle='-', linewidth=1)
        axs.set_title("Weights of the first layer")
        axs.tick_params(which="minor", bottom=False, left=False)
        axs.tick_params(which="major", bottom=False, left=True)

        axs.set_xticks([])
        axs.set_yticks([i for i in range(ndim)])
        axs.set_yticklabels([fr"$x_{{{i+1}}}$" for i in range(ndim)])

        # axs[1].imshow(np.absolute(par_list[1][1])>0)
        # axs[1].set_title(par_list[1][0])
        # axs[1].set_yticks([])


# for n, (ax, model_id) in enumerate(zip(axs, model_ids)):
#     # get all info from saved dict
#     model_data = torch.load(io_path.model / f'{model_id}.pt')
#     model_arch = model_data['info']['architecture']
#     model_args = model_data['info']['arguments']
#     state_dict = remove_mask(model_data['best']['model_dict'])
#
#     model = getattr(models, model_arch)(**model_args)
#     model.load_state_dict(state_dict)
#     model.eval()
#
#     par_dict = {}
#     for coder_name, coder in model.named_children():
#         for name, par in coder.named_parameters():
#             par_np = par.detach().numpy()
#             if par_np.ndim == 1:
#                 par_np = par_np[:, np.newaxis]
#             layer_name, par_name = name.split('.')
#             value = par_dict.setdefault(coder_name+layer_name, [])
#
#             value.append((par_name, par_np.T))
#
#     clim = (0, 1)
#     # plt.setp(axs, xticks=[], yticks=[])
#     for n, (layer_name, par_list) in enumerate(par_dict.items()):
#         if n == 0:
#             ax.imshow(np.absolute(par_list[0][1])>0, clim=clim)
#             ax.set_title("Model 2")


# ax.set_xlabel('Models')
# ax.set_ylabel('Error')

# plt.tight_layout()
plt.savefig(io_path.figs / f"{script_name}.pdf")
plt.close()
