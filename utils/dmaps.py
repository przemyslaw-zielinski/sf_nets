#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 10 Feb 2020

@author: Przemyslaw Zielinski
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import utils.spaths as spaths
from scipy.linalg import sqrtm
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

def lnc_ito(data, sde):

    disp_val = sde.disp(0, data)

    return np.einsum('bij,bkj->bik', disp_val, disp_val)

def ln_covs(data, sde, solver, nsam, dt,
            nsteps=1, tqdm_off=True, batch_size=None):
    """
    Takes data array of shape (N, D) and returns an array of sample local
    noise covariances of shape (N, D, D)
    """

    data_rep = np.repeat(data.astype(dtype=np.float32), nsam, axis=0)
    batch = solver.burst(sde, data_rep, (0.0, nsteps), dt)

    covs = []
    fact = nsam - 1
    for point_batch in np.split(batch, len(data)):
        point_batch -= np.average(point_batch, axis=0)
        covs.append(point_batch.T @ point_batch / (dt * fact))

    return np.array(covs)#.astype(dtype=np.float32)

def data_affinity(data, covariances, epsilon):
    precisions = [np.linalg.pinv(cov) for cov in covariances]
    npoints, dim = data.shape
    weights = np.zeros((npoints, npoints))
    print(f"Computing affinity matrix for epsilon = {epsilon} ")
    for i1, point1 in enumerate(data):
        for j, point2 in enumerate(data[i1:]):
            i2 = i1 + j
            sum_prec = precisions[i1] + precisions[i2]
            diff_points = point2 - point1
            quad_form = np.dot(diff_points, sum_prec @ diff_points) / 2
            weights[i1, i2] = np.exp(- quad_form /  epsilon**2)
            weights[i2, i1] = weights[i1, i2]
    return weights

def affinity(data, sde, nsam, dt, epsilon, rng, transform=None):

    bcovs = ln_covs(data, sde, nsam, dt, rng, transform=transform)
    return data_affinity(data, bcovs, epsilon)

def markov_eig(W, norm0=1):
    # compute similar symmetric matrix
    Dinv = np.linalg.inv(np.diag(np.sum(W, axis=1)))
    S = sqrtm(Dinv) @ W @ sqrtm(Dinv)

    eigvals, eigvecs = np.linalg.eigh(S)
    eigvecs = sqrtm(Dinv) @ eigvecs # transf to eigvecs of A

    # sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs.T[order]

    # normalize such that eigvecs[0] = [1,1,...,1]
    eignorms = np.linalg.norm(eigvecs, axis=1, keepdims=True)
    eigvecs = norm0 * eigvecs / eignorms
    if np.average(eigvecs[0]) < 0:
        # print("Changing sign.")
        eigvecs = -1 * eigvecs

    return eigvals, eigvecs

def adm(data, sde, batch_nsam, batch_dt, epsilon, rng,
        nsteps=1, transform=None, norm0=1):

    bcovs = ln_covs(data, sde, batch_nsam, batch_dt, rng,
                       nsteps=nsteps, transform=transform)
    weight_matrix = data_affinity(data, bcovs, epsilon)
    eigvals, eigvecs = markov_eig(weight_matrix, norm0=norm0)

    return eigvals, eigvecs, weight_matrix


def test_epsilon(data, sde, dt, rng, logspace_params, bnsam=10**4):
    from IPython.display import clear_output
    import time

    eps_vals = np.logspace(*logspace_params)
    L = []
    for n, eps in enumerate(eps_vals):
        print(f"Run: {n+1}")
        time.sleep(.5)
        W = affinity(data, sde, bnsam, dt, eps, rng)
        clear_output()
        L.append(np.sum(W))
    return eps_vals, L

def test_eigs(A, eigvals, eigvecs):
    # corresponding eigenvectors are now in the rows of eigenvecs matrix
    allclose = True
    for eigval, eigvec in zip(eigvals, eigvecs):
        allclose = allclose and np.allclose(A @ eigvec, eigval * eigvec)
    return allclose

def poly_features(data, names, max_deg=1, const=False):
    import itertools

    powers = itertools.product(range(max_deg + 1), repeat=len(names))
    if not const: next(powers)

    X = {}
    for p in powers:
        p = np.array(p)
        if np.sum(p) > max_deg:
            continue
        nonz = p > 0
        names_nonz = itertools.compress(names, nonz)
        p_nonz = p[nonz]
        b = "".join([f"{var}^{i}" for var, i in zip(names_nonz, p_nonz)])
        X[b] = np.prod(data**p, axis=1) # x**n * y**m
    return pd.DataFrame(X)

def lasso_adm(X, Y, alpha_list, rng, test_size=0.3, max_iter=10**5):

    seed = rng.integers(1000)
    X_train,X_test,y_train,y_test=train_test_split(X, Y,
                                                   test_size=test_size,
                                                   random_state=seed)

    results = {}
    index = list(X.columns) + ["train_score", "test_score"]
    for alpha in alpha_list:
        lasso = Lasso(alpha=alpha, max_iter=max_iter)
        lasso.fit(X_train,y_train)
        scores = [lasso.score(X_train,y_train), lasso.score(X_test,y_test)]
        results[alpha] = np.concatenate((lasso.coef_, scores))
    return pd.DataFrame(results, index=index)
