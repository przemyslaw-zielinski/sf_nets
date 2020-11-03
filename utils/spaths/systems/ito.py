#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 2020

@author: Przemyslaw Zielinski
"""

import numpy as np
import jax.numpy as jnp
from inspect import signature
from jax import jacfwd, hessian, jit, vmap

class ItoSDE():

    def __init__(self, drift, dispersion, noise_mixing_dim=0):
        '''
        Ito stochastic differential equation

            dX = A(t, X)dt + B(t, X)dW

        Parameters 'drift' and 'dispersion' have to be functions of
        scalar time t and arrays x, and optionally of array dx,
        such that dx stores the value of A and B respectively:

        ->  def drift(t, x):
                return A(t, x)
        ->  def drift(t, x, dx):
                dx[:] = A(t, x)

        ->  dispersion(t, x):
                return B(t, x)
        ->  dispersion(t, x, dx):
                dx[:] = B(t, x)

        Here t is float and x.shape = (ndim, nsam).
        The drift should result in an array of shape (ndim, nsam).
        The dispersion should result in an array of shape (ndim[, nmd], nsam)

        Parameters
        ----------
            drift, diffusion (callable) : coefficients of the equation
            noise_mixing_dim (int) : if positive, the dimension of noise
                if zero, indicates the diagonal noise
        '''

        self._drif = drift
        self._disp = dispersion

        self.drif = drift if is_explicit(drift) else self._im_drif
        self.disp = dispersion if is_explicit(dispersion) else self._im_disp

        if noise_mixing_dim == 0:  # diagonal noise
            self.nmd = ()
            self.dnp = self._diag_dnp
            self.diff = self._diag_diff
        else:
            self.nmd = (noise_mixing_dim,)
            self.dnp = self._gene_dnp
            self.diff = self._gene_diff

    # drift and dispersion wrappers when they use dx array
    # here x.shape = (d, s)
    def _im_drif(self, t, x):
        dx = np.zeros_like(x)
        self._drif(t, x, dx)
        return dx

    def _im_disp(self, t, x):
        dx = np.zeros(self.get_disp_shape(x), dtype=x.dtype)
        self._disp(t, x, dx)
        return dx

    # TODO: can we use that in solvers?
    # def coeffs(self, t, x):
    #     return self.drif(t, x), self.disp(t, x)

    # options for diffusion computations depending on noise type
    # _diag = diagonal noise (nmd=0), _gene = general noise (nmd>0)
    def _diag_diff(self, t, x):
        disp = self.disp(t, x)  # disp.shape = (d, s)
        diff = disp * disp
        # multiplies dXd identity matrix by each col of diff via broadcasting
        return np.eye(x.shape[0])[..., np.newaxis] * diff

    def _gene_diff(self, t, x):
        disp = self.disp(t, x)  # disp.shape = (d, m, s)
        # multiply disp by its transpose with batching along last axis
        return np.einsum('ijs, kjs->iks', disp, disp)

    # options to compute dispersion noise product
    def _diag_dnp(self, t, x, dw):
        return self.disp(t, x) * dw

    def _gene_dnp(self, t, x, dw):
        return np.einsum('dms,ms->ds', self.disp(t, x), dw)

    ##########################################################
    # versions for computations with ensembles (= data arrays)
    def ens_drif(self, t, ens):
        return self.drif(t, ens.T).T

    def ens_disp(self, t, ens):
        # moveaxis: (d[, m], s) -> (s, d[, m])
        return np.moveaxis(self.disp(t, ens.T), -1, 0)

    def ens_diff(self, t, ens):
        # moveaxis: (d, d, s) -> (s, d, d)
        return np.moveaxis(self.diff(t, ens.T), -1, 0)

    def ens_dnp(self, t, ens, ens_dw):
        return self.dnp(t, ens.T, ens_dw.T).T
    ##########################################################

    def _test_dim(self, ens):
        if ens.ndim != 2 or ens.shape[1] != self.ndim:
            raise IndexError(f"Bad ensemble: shape={ens.shape}.")

    def get_noise_shape(self, ens):
        return ens.shape if self.nmd == () else ens.shape[:1] + self.nmd

    def get_disp_shape(self, x):
        return x.shape if self.nmd == () else (x.shape[0], self.nmd[0], x.shape[1])


def is_explicit(coeff_func):
    sig = signature(coeff_func)
    return len(sig.parameters) == 2

class SDETransform():

    def __init__(self, func, ifunc):
        '''
        function(x) = y with x.shape = (d, b), y.shape = (p, b)
        where d - input dimesion, p - output dimension, b - batch dimension
        (coord major)
        '''

        self.f = jit(func)
        self.g = jit(ifunc)

        self.df = jit(vmap(jacfwd(func), in_axes=1, out_axes=2))
        # we map over the batch dimension b
        # (d, b) -> (p, d, b): array of gradients of components of function
        self.ddf = jit(vmap(hessian(func), in_axes=1, out_axes=3))
        # (d, b) -> (p, d, d, b)

    def __call__(self, sde):
        nmd = sde.nmd[0] if sde.nmd else 0  # TODO: diag noise can change into non-diag
        return ItoSDE(self.func_drif(sde), self.func_disp(sde), noise_mixing_dim=nmd)

    def func_drif(self, sde):

        def drif(t, y):
            x = self.g(y)
            return (
                batch_dot(self.df(x), sde.drif(t, x)) + \
                batch_trace(batch_quad(self.ddf(x), sde.disp(t, x))) / 2
                )
        return drif

    def func_disp(self, sde):

        def disp(t, y):
            x = self.g(y)
            return batch_mul(self.df(x), sde.disp(t, x))

        return disp

def batch_dot(bmat, bvec):
    return np.einsum('pdb,db->pb', bmat, bvec)

def batch_mul(bmat1, bmat2):
    return np.einsum('pdb,dmb->pmb', bmat1, bmat2)

def batch_quad(bten, bmat):
    bmatt = np.moveaxis(bmat, 1, 0)
    return np.einsum('mdb,pdcb,cnb->pmnb', bmatt, bten, bmat)

def batch_trace(bten):
    return np.einsum('pmmb->pb', bten)
