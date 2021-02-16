"""
Created on Tue 16 Feb 2021

@author: Przemyslaw Zielinski
"""

import torch
from math import sqrt

def fast_ortho(model, dataset):

    sdim = dataset.system.sdim
    ndim = dataset.system.ndim
    fdim = ndim - sdim

    data = dataset.data.requires_grad_(True)
    stbs = torch.eye(sdim).repeat(len(data), 1, 1).T  # standard basis
    prcs = dataset.precs

    evals, evecs = torch.symeig(prcs, eigenvectors=True)
    f_evecs = evecs[:, :, :fdim]

    norm_grads = []
    for ei in stbs:
        v = model.encoder(data)
        v.backward(ei.T)

        grad = data.grad.clone()
        data.grad.zero_()

        norm = torch.linalg.norm(grad, dim=1, keepdim=True)
        norm_grads.append(grad / norm)
    norm_grads = torch.stack(norm_grads, dim=2)

    U = torch.vstack([f_evecs.T, norm_grads.T]).T
    UT = torch.transpose(U, 1, 2)
    UTU = torch.matmul(UT, U)

    return torch.linalg.norm(UTU - torch.eye(ndim), dim=(1,2)) / sqrt(ndim)
