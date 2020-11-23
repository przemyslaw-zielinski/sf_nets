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
from matplotlib import pyplot as plt
from utils.mpl_utils import scale_figsize

# matplotlib settings
plt.style.use("utils/manuscript.mplstyle")
cdata, cslow, cfast = 'C0', 'C1', 'C2'  # colors

dat_train, lnc_train, _ = torch.load(data_path / 'processed' / 'train.pt')
dat_test, lnc_test, _ = torch.load(data_path / 'processed' / 'test.pt')

data = np.vstack([dat_train, dat_test])
ln_covs = np.vstack([lnc_train, lnc_test])

e_vals, e_vecs = zip(*[np.linalg.eigh(cov) for cov in ln_covs])
e_vals, e_vecs = np.array(e_vals), np.array(e_vecs)

step = 15  # skip some data points
fig, axs = plt.subplots(ncols=2, figsize=scale_figsize(width=4/3))

# eigenvalues for a subset of dataset
e0, e1 = e_vals[::step].T
axs[0].plot(e0, 'o', label=r"slow", c=cslow, zorder=1)
axs[0].plot(e1, 'o', label=r"fast", c=cfast, zorder=2)
axs[0].legend()
axs[0].set_yscale('log')

axs[0].set_xlabel('data point')
axs[0].set_title('Eigenvalues of local noise covariances')

# eigenvectors
X, Y = data[::step].T
XS, YS = zip(*[e_vec[:,0] for e_vec in e_vecs[::step]])
XF, YF = zip(*[e_vec[:,1] for e_vec in e_vecs[::step]])

# axs[1].scatter(*data[::step].T, alpha=.5, s=5)
axs[1].quiver(X, Y, XS, YS, label="slow",
              # pivot="middle", #width=0.005,
              # headwidth=0, headlength=0, headaxislength=0,  # remove arrowhead
              scale=20,
              color=cslow)
axs[1].quiver(X, Y, XF, YF, label="fast",
              # pivot="middle", #width=0.005,
              # headwidth=0, headlength=0, headaxislength=0,  # remove arrowhead
              scale=20,
              color=cfast)

leg = axs[1].legend()
# for line in leg.get_patches():
#     line.set_lw(.01)

axs[1].set_title("Eigenvectors of local noise covariances")
axs[1].set_xlabel(r"$x^1$",)
axs[1].set_ylabel(r"$x^2$", rotation=0)

axs[1].set_xlim([0,2*np.pi])
axs[1].set_ylim([-3.5,3.5])
axs[1].set_xticks([0, np.pi, 2*np.pi])
axs[1].set_xticklabels(['0', r'$\pi$', r'$2\pi$'])

fig.tight_layout()
plt.savefig(figs_path / 'sin2d_lnc_eigendecomp.pdf')
plt.close(fig)
