#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 17 Nov 2020

@author: Przemyslaw Zielinski
"""

import sys
from pathlib import Path
root = Path.cwd()
sys.path[0] = str(root)
data_path = root / 'data' / 'Sin2'
figs_path = root / 'results' / 'figs' / 'sin2d'

import torch
import numpy as np
import matplotlib as mpl
import sf_nets.datasets as datasets
from matplotlib import pyplot as plt
from utils.mpl_utils import scale_figsize

# matplotlib settings
plt.style.use("utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors
PI = np.pi

def to_darray(*meshgrids):
    return np.stack(meshgrids).reshape(len(meshgrids), -1).T

def to_grid(darray, grid_size):
    if darray.ndim == 1:
        return darray.reshape(grid_size, -1)
    else:
        return darray.reshape(darray.shape[1], grid_size, -1)

dat_train, lnc_train, _ = torch.load(data_path / 'processed' / 'train.pt')
dat_test, lnc_test, _ = torch.load(data_path / 'processed' / 'test.pt')

dataset = datasets.Sin2(root / 'data')
slow_map = dataset.system.slow_map

data = np.vstack([dat_train, dat_test])
ln_covs = np.vstack([lnc_train, lnc_test])

e_vals, e_vecs = zip(*[np.linalg.eigh(cov) for cov in ln_covs])
e_vals, e_vecs = np.array(e_vals), np.array(e_vecs)

step = 25  # skip some data points
fig, axs = plt.subplots(ncols=2, figsize=scale_figsize(width=4/3))

# eigenvalues for a subset of dataset
e1, e2 = e_vals.T
axs[0].boxplot(e1, sym='', labels=[f'eigenvalue 1'],
    positions=[0],
    medianprops={'color': cslow, 'lw': 2})

axs[0].boxplot(e2, sym='', labels=[f'eigenvalue 2'],
    positions=[1],
    medianprops={'color': cfast, 'lw': 2})

# axs[0].plot(e0, 'o', label=r"slow", c=cslow, zorder=1)
# axs[0].plot(e1, 'o', label=r"fast", c=cfast, zorder=2)
# axs[0].legend()
axs[0].set_yscale('log')

# axs[0].set_xlabel('data point')
axs[0].set_title('Eigenvalues of local noise covariances')

# fast fbers
mesh_size = 400
x = np.linspace(+0.0, 2*PI, mesh_size)
y = np.linspace(-PI, PI, mesh_size)
X, Y = np.meshgrid(x, y)

mesh_data = to_darray(X, Y)
v = slow_map(mesh_data.T).T
V = np.squeeze(to_grid(v, mesh_size))

axs[1].contour(X, Y, V, levels=10, colors=cfast, linewidths=.5, alpha=.8)

# eigenvectors
X, Y = data[::step].T
XS, YS = zip(*[e_vec[:,0] for e_vec in e_vecs[::step]])
XF, YF = zip(*[e_vec[:,1] for e_vec in e_vecs[::step]])

# axs[1].scatter(*data[::step].T, alpha=.5, s=5)
axs[1].quiver(X, Y, XS, YS, label="slow",
              # pivot="middle", #width=0.005,
              # headwidth=0, headlength=0, headaxislength=0,  # remove arrowhead
              scale=20,
              color=cslow,
              zorder=3)
axs[1].quiver(X, Y, XF, YF, label="fast",
              # pivot="middle", #width=0.005,
              # headwidth=0, headlength=0, headaxislength=0,  # remove arrowhead
              scale=20,
              color=cfast,
              zorder=3)

leg = axs[1].legend()
# for line in leg.get_patches():
#     line.set_lw(.01)

axs[1].set_title("Eigenvectors of local noise covariances")
axs[1].set_xlabel(r"$x^1$",)
axs[1].set_ylabel(r"$x^2$", rotation=0)

axs[1].set_xlim([0, 2*PI])
axs[1].set_ylim([-PI, PI])
axs[1].set_xticks([0, PI, 2*PI])
axs[1].set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
axs[1].set_aspect('equal')

fig.tight_layout()
plt.savefig(figs_path / 'sin2d_lnc_eigendecomp.pdf')
plt.close(fig)
