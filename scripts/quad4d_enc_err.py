#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 4 Feb 2021

@author: Przemyslaw Zielinski
"""

script_name = "quad4d_enc_err"

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
from utils.mpl_utils import scale_figsize
from utils import spaths

# matplotlib settings
plt.style.use("utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors

name_ds = 'Quad4'
model_type = "mahl1_elu"

data_path = root / 'data' / name_ds
figs_path = root / 'results' / 'figs' / f"{name_ds.lower()}d"
model_path = root / 'results' / 'models' / name_ds

dataset = getattr(datasets, name_ds)(root / 'data', train=False)  # use test ds
slow_map = dataset.system.slow_map
slow_map_der = dataset.system.slow_map_der
figs_path.mkdir(exist_ok=True)

dat_t = dataset.data
dat_np = dat_t.detach().numpy()

test_precs = dataset.precs
test_evals, test_evecs = torch.symeig(test_precs, eigenvectors=True)
f_evecs = test_evecs[:, :, :3]

model_ids = [f"mse_elu_{n+1}" for n in range(3)]

### Derivatives ###

dat_t.requires_grad_(True);
g = torch.eye(1).repeat(len(dat_t),1,1).T
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

    grads_norm = []
    for gi in g:
        v = model.encoder(dat_t)
        v.backward(gi.T)
        grad = dat_t.grad.clone()
        grads_norm.append(grad / torch.linalg.norm(grad, dim=1, keepdim=True))
        dat_t.grad.zero_();
    grads_norm = torch.stack(grads_norm, dim=2)

    A = torch.vstack([f_evecs.T, grads_norm.T]).T
    AT = torch.transpose(A, 1, 2)
    AAT = torch.matmul(AT, A)
    diff = torch.linalg.norm(AAT - np.eye(4), dim=(1,2))
    diff = diff.numpy()

    ax.boxplot(diff, positions=[n], sym='', labels=[f'model {n+1}'])

der = torch.from_numpy(slow_map_der(dat_np.T).T).float()
der_norm = der / torch.linalg.norm(der, dim=1, keepdim=True)
B = torch.vstack([f_evecs.T, der_norm.T]).T
BT = torch.transpose(B, 1, 2)
BBT = torch.matmul(BT, B)

diff = torch.linalg.norm(BBT - torch.eye(4), dim=(1,2))
diff = diff.numpy()

ax.axvline(n+.6, ls='--', c='k', alpha=.5)
# slow_vars = np.var(slow_map(bursts.T).T, axis=1) / dt
ax.boxplot(diff, positions=[n+1], sym='', labels=['slow variable'])

ax.set_xlabel('Models')
ax.set_ylabel('Error')

plt.tight_layout()
plt.savefig(figs_path / f"{script_name}_derivatives.pdf")
plt.close()


### FIBERS ###
nreps = 20_000
tsep = dataset.eps
ndim = dataset.system.ndim
dt = dataset.burst_dt / 5

sde = dataset.system

# seed setting
seed = 3579
rng = np.random.default_rng(seed)
rng.integers(10**3);  # warm up of RNG

# stochastic solver
em = spaths.EulerMaruyama(rng)

fibs = em.solve(sde, dat_np, (0, 0.5*tsep), dt).p
fibs = torch.from_numpy(fibs).float()

dat_t.requires_grad_(False)
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

    with torch.no_grad():
        vfibs =  model.encoder(fibs)
        vimg =  model.encoder(dat_t)
    var_vfibs = torch.var(vfibs, axis=1)
    norm_var_vfibs = var_vfibs / torch.mean(torch.var(vimg, axis=0))
    norm_var_vfibs = norm_var_vfibs.numpy()

    ax.boxplot(norm_var_vfibs, positions=[n], sym='', labels=[f'model {n+1}'])


sfibs = slow_map(fibs.T).T
simg = slow_map(dat_t.T).T

var_sfibs = torch.var(sfibs, axis=1)
norm_var_sfibs = var_sfibs / torch.mean(torch.var(simg, axis=0))
norm_var_sfibs = norm_var_sfibs.numpy()

ax.axvline(n+.6, ls='--', c='k', alpha=.5)
# slow_vars = np.var(slow_map(bursts.T).T, axis=1) / dt
ax.boxplot(norm_var_sfibs, positions=[n+1], sym='', labels=['slow variable'])

ax.set_xlabel('Models')
ax.set_ylabel('Error')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(figs_path / f"{script_name}_fibers.pdf")
plt.close()
