#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 26 Feb 2022

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

import spaths
from sf_nets.utils.mpl_utils import scale_figsize
from sf_nets.utils.io_utils import io_path, get_script_name

ds_base_name = 'Quad4'
eps_variants = ['0_001', '0_0031', '0_01', '0_031', '0_1', '0_31', '1_0']
model_ids = ['mse_elu_3'] + [f'mse_elu_3_r{n}' for n in range(1, 10)]
print(model_ids)

script_name = get_script_name()
base_path = io_path(ds_base_name)

# matplotlib settings
plt.style.use("sf_nets/utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors

model_id = "mse_elu_3"
ds_names = [f'{ds_base_name}_{eps}' for eps in eps_variants]

avgs, stds = [], []
fig, ax = plt.subplots(figsize=scale_figsize(height=.9)) #figsize=scale_figsize(width=4/3))
for n, (ds_name, eps) in enumerate(zip(ds_names, eps_variants)):
    path = io_path(ds_name)
    
    # get all info from saved dict
    dataset = getattr(datasets, ds_name)(path.dataroot, train=False)  # use test ds
    slow_map = dataset.system.slow_map
    slow_map_der = dataset.system.slow_map_der

    dat_t = dataset.data
    dat_np = dat_t.detach().numpy()

    test_precs = dataset.precs
    test_evals, test_evecs = torch.symeig(test_precs, eigenvectors=True)
    f_evecs = test_evecs[:, :, :3]

    dat_t.requires_grad_(True)
    g = torch.eye(1).repeat(len(dat_t),1,1).T

    avg_per_eps = []
    std_per_eps = []
    for model_id in model_ids:
        model_data = torch.load(path.models / f'{model_id}.pt')
        model_arch = model_data['info']['architecture']
        model_args = model_data['info']['arguments']
        state_dict = model_data['best']['model_dict']

        model = getattr(models, model_arch)(**model_args)
        model.load_state_dict(state_dict)
        model.eval()

        grads_norm = []
        for gi in g:
            v = model.encoder(dat_t)
            v.backward(gi.T)
            grad = dat_t.grad.clone()
            grads_norm.append(grad / torch.linalg.norm(grad, dim=1, keepdim=True))
            dat_t.grad.zero_()
        grads_norm = torch.stack(grads_norm, dim=2)

        A = torch.vstack([f_evecs.T, grads_norm.T]).T
        AT = torch.transpose(A, 1, 2)
        AAT = torch.matmul(AT, A)
        diff = torch.linalg.norm(AAT - np.eye(4), dim=(1,2))
        diff = diff.numpy()
        avg_per_eps.append(diff.mean())
        std_per_eps.append(diff.std())

    avgs.append(np.mean(avg_per_eps))
    stds.append(np.mean(std_per_eps))
avgs = np.array(avgs)
stds = np.array(stds)

    # ax.boxplot(
    #     diff,
    #     positions=[n],
    #     sym='',
    #     labels=[f'{float(eps.replace("_","."))}']
    # )
eps_float = [float(f'{eps.replace("_",".")}') for eps in eps_variants]
print(eps_float)

ax.fill_between(eps_float, avgs + stds, avgs - stds, color=cdata, alpha=0.2)
ax.plot(eps_float, avgs, color=cslow)

ax.set_xscale('log')
ax.set_xlim([0.001, 1.0])
ax.set_xticks([0.001, 0.01, 0.1, 1.0])
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel('Error')

plt.tight_layout()
plt.savefig(base_path.figs / f"{script_name}_derivatives.pdf")
plt.close()