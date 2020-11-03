#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 2020

@author: Przemyslaw Zielinski
"""

import numpy as np
from .ito import ItoSDE
from jax import grad, vmap, jit  # TODO: catch potential import error
from ..potentials import PairwisePotential

class OverdampedLangevin(ItoSDE):
    '''
    dX = -gradV(t,X)dt + sqrt(2*inv_temp**(-1))dW
    '''
    def __init__(self, V, inv_temp):

        if isinstance(V, PairwisePotential):
            self.gradV = V.grad
        else:
            _V = squeeze(V)
            self.gradV = jit(vmap(grad(_V, 1), in_axes=(None, 1), out_axes=1))
        # settings for vmap
        # in_axes=(None, 1): don't parallelize over the time and parallelize
        #                    over the samples axis
        # out_axes=1: put result along second axis
        self.inv_temp = inv_temp

        super().__init__(self.ol_drift, self.ol_dispersion)

    def ol_drift(self, t, u):#, du):
        # du[:] = -self.gradV(t, u)
        return -self.gradV(t, u)

    def ol_dispersion(self, t, u):#, du):
        # du[:] = np.sqrt(2 / self.inv_temp)
        return np.sqrt(2 / self.inv_temp) * np.ones_like(u)

def squeeze(func):
    def wrapper(*args, **kwargs):
        return np.squeeze(func(*args, **kwargs))
    return wrapper
