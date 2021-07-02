#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 15 Dec 2020

@author: Przemyslaw Zielinski
"""

import sys
from pathlib import Path
root = Path.cwd()
sys.path[0] = str(root)

import torch
import numpy as np
import matplotlib as mpl
import sf_nets.models as models
import sf_nets.datasets as datasets
from matplotlib import pyplot as plt
from utils.mpl_utils import scale_figsize
import spaths

# matplotlib settings
plt.style.use("utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors

name_ds = 'Quad4'
model_type = "mahl1_elu"

data_path = root / 'data' / name_ds
figs_path = root / 'results' / 'figs' / name_ds.lower()
model_path = root / 'results' / 'models' / name_ds

dataset = getattr(datasets, name_ds)(root / 'data', train=False)  # use test ds
slow_map = dataset.system.slow_map
figs_path.mkdir(exist_ok=True)

dat_t = dataset.data
dat_np = dat_t.detach().numpy()

model_ids = [model_type + f"_{n}" for n in range(3)]

nreps = 20_000
tsep = dataset.eps
ndim = dataset.system.ndim
dt = dataset.burst_dt / 10

sde = dataset.system

# seed setting
seed = 3579
rng = np.random.default_rng(seed)
rng.integers(10**3);  # warm up of RNG

# stochastic solver
em = spaths.EulerMaruyama(rng)

dp = rng.permutation(dat_np)
ens0 = np.repeat(dp, nreps, axis=0)
bursts = em.burst(sde, ens0, (0, 1), dt).reshape(len(dp), nreps, ndim)
bursts_t = torch.from_numpy(bursts).to(dtype=torch.float)

fig, ax = plt.subplots(figsize=scale_figsize(width=4/3))
for n, model_id in enumerate(model_ids):
    # get all info from saved dict
    model_data = torch.load(model_path / f'{model_id}.pt')
    model_arch = model_data['info']['architecture']
    model_args = model_data['info']['arguments']
    state_dict = model_data['best']['model_dict']

    model = getattr(models, model_arch)(**model_args)
    model.load_state_dict(state_dict)
    model.eval()

    slow_view = model.encoder(bursts_t).detach().numpy()
    slow_view_vars = np.var(slow_view, axis=1) / dt

    ax.boxplot(slow_view_vars, positions=[n-.2], sym='', labels=[f'{n+1}a'])

    model_data = torch.load(model_path / f'{model_id}a.pt')
    model_arch = model_data['info']['architecture']
    model_args = model_data['info']['arguments']
    state_dict = model_data['best']['model_dict']

    model = getattr(models, model_arch)(**model_args)
    model.load_state_dict(state_dict)
    model.eval()


    slow_view = model.encoder(bursts_t).detach().numpy()
    slow_view_vars = np.var(slow_view, axis=1) / dt

    ax.boxplot(slow_view_vars, positions=[n+.2], sym='', labels=[f'{n+1}b'])

# ax.set_ylim([0, 100])

ax.axvline(n+.6, ls='--', c='k', alpha=.5)
slow_vars = np.var(slow_map(bursts.T).T, axis=1) / dt
ax.boxplot(slow_vars, positions=[n+1], sym='', labels=['slow variable'])

ax.set_xlabel('Models')
ax.set_ylabel('Variance')

plt.tight_layout()
plt.savefig(figs_path / f"{name_ds.lower()}_{model_type}_slowview_var.pdf")
plt.close()
