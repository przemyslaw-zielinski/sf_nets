"""
Created on Tue 16 Feb 2021

@author: Przemyslaw Zielinski
"""

import torch
from math import sqrt

def fast_ortho(encdec_model, dataset):

    sdim = dataset.system.sdim
    ndim = dataset.system.ndim
    fdim = ndim - sdim

    data = dataset.data#.requires_grad_(True)
    stbs = torch.eye(sdim).repeat(len(data), 1, 1).T  # standard basis
    prcs = dataset.precs

    evals, evecs = torch.symeig(prcs, eigenvectors=True)
    f_evecs = evecs[:, :, :fdim]

    return ortho_error(encdec_model.encoder, data, f_evecs)

    # norm_grads = []
    # for ei in stbs:
    #     v = model.encoder(data)
    #     v.backward(ei.T)
    #
    #     grad = data.grad.clone()
    #     data.grad.zero_()
    #
    #     norm = torch.linalg.norm(grad, dim=1, keepdim=True)
    #     norm_grads.append(grad / norm)
    # norm_grads = torch.stack(norm_grads, dim=2)
    #
    # U = torch.vstack([f_evecs.T, norm_grads.T]).T
    # UT = torch.transpose(U, 1, 2)
    # UTU = torch.matmul(UT, U)
    #
    # return torch.linalg.norm(UTU - torch.eye(ndim), dim=(1,2)) / sqrt(ndim)

def ortho_error(model, data, vecs):
    """
    model: with s output dim
    data: (b, d)
    vecs: (b, d, f)
    """

    # ndim = data.size()[1]
    # fdim = vecs.size()[2]
    # sdim = ndim - fdim
    b, d = data.size()

    data.requires_grad_(True)
    v = model(data)

    stbs = torch.eye(v.size()[1]).repeat(b, 1, 1).T  # standard basis

    norm_grads = []
    for ei in stbs:

        v.backward(ei.T, retain_graph=True)

        grad = data.grad.clone()
        data.grad.zero_()

        norm = torch.linalg.norm(grad, dim=1, keepdim=True)
        norm_grads.append(grad / norm)
    norm_grads = torch.stack(norm_grads, dim=2)

    U = torch.vstack([vecs.T, norm_grads.T]).T
    UT = torch.transpose(U, 1, 2)
    UTU = torch.matmul(UT, U)
    I = torch.eye(U.size()[2])

    return torch.linalg.norm(UTU - I, dim=(1,2)) / sqrt(d)
