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
import sf_nets.models as models
import sf_nets.datasets as datasets
from sf_nets.metrics import fast_ortho, ortho_error
from matplotlib import pyplot as plt
from sf_nets.utils.mpl_utils import scale_figsize
from sf_nets.utils.io_utils import io_path
import spaths

ds_name = 'Quad10'
script_name = "quad10d_enc_err"
io_path = io_path(ds_name)

# matplotlib settings
plt.style.use("sf_nets/utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors

dataset = getattr(datasets, ds_name)(io_path.dataroot, train=False)  # use test ds

with open("experiments.yaml", 'rt') as yaml_file:
    models_data = yaml.full_load(yaml_file)[ds_name]
model_ids = models_data['model_ids']
model_tags = models_data['model_tags']

fig, ax = plt.subplots(figsize=scale_figsize(width=4/3))
for n, (id, tag) in enumerate(zip(model_ids, model_tags)):
    # get all info from saved dict
    model_data = torch.load(io_path.models / f'{id}.pt')
    model_arch = model_data['info']['architecture']
    model_args = model_data['info']['arguments']
    state_dict = model_data['best']['model_dict']

    model = getattr(models, model_arch)(**model_args)
    model.load_state_dict(state_dict)
    model.eval()

    label = tag.split(' ')[-1]

    diff = fast_ortho(model, dataset)
    if 'pruned' in id:
        position = n - .7
    else:
        position = n

    ax.boxplot(diff.numpy(), positions=[position], sym='', labels=[label])

ax.set_xlabel('Models')
ax.set_ylabel('Error')

plt.tight_layout()
plt.savefig(io_path.figs / f"{script_name}_derivatives.pdf")
plt.close()


# ### FIBERS ###
# nreps = 20_000
# tsep = dataset.eps
# ndim = dataset.system.ndim
# dt = dataset.burst_dt / 5
#
# sde = dataset.system
#
# # seed setting
# seed = 3579
# rng = np.random.default_rng(seed)
# rng.integers(10**3);  # warm up of RNG
#
# # stochastic solver
# em = spaths.EulerMaruyama(rng)
#
# fibs = em.solve(sde, dat_np, (0, 0.5*tsep), dt).p
# fibs = torch.from_numpy(fibs).float()
#
# dat_t.requires_grad_(False)
# fig, ax = plt.subplots(figsize=scale_figsize(width=4/3))
# for n, model_id in enumerate(model_ids):
#     # get all info from saved dict
#     model_data = torch.load(path.models / f'{model_id}.pt')
#     model_arch = model_data['info']['architecture']
#     model_args = model_data['info']['arguments']
#     state_dict = remove_mask(model_data['best']['model_dict'])
#
#     model = getattr(models, model_arch)(**model_args)
#     model.load_state_dict(state_dict)
#     model.eval()
#
#     with torch.no_grad():
#         vfibs =  model.encoder(fibs)
#         vimg =  model.encoder(dat_t)
#     var_vfibs = torch.var(vfibs, axis=1)
#     norm_var_vfibs = var_vfibs / torch.mean(torch.var(vimg, axis=0))
#     norm_var_vfibs = norm_var_vfibs.numpy()
#
#     position = n - .7 if 'pruned' in model_id else n
#     label = f'{n}p' if 'pruned' in model_id else f'{n+1}'
#     ax.boxplot(norm_var_vfibs.mean(1), positions=[position], sym='', labels=[label])
#
#
# sfibs = slow_map(fibs.T).T
# simg = slow_map(dat_t.T).T
#
# var_sfibs = torch.var(sfibs, axis=1)
# norm_var_sfibs = var_sfibs / torch.mean(torch.var(simg, axis=0))
# norm_var_sfibs = norm_var_sfibs.numpy()
#
# ax.axvline(n+.6, ls='--', c='k', alpha=.5)
# # slow_vars = np.var(slow_map(bursts.T).T, axis=1) / dt
# ax.boxplot(norm_var_sfibs.mean(1), positions=[n+1], sym='', labels=['slow variable'])
#
# ax.set_xlabel('Models')
# ax.set_ylabel('Error')
# ax.set_yscale('log')
#
# plt.tight_layout()
# plt.savefig(path.figs / f"{script_name}_fibers.pdf")
# plt.close()
