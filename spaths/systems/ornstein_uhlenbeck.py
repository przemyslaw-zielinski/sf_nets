#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 3 Apr 2020

@author: Przemyslaw Zielinski
"""

import numpy as np
from .ito import ItoSDE
from numpy.linalg import eig, inv
from scipy.integrate import quad

class OrnsteinUhlenbeck(ItoSDE):
    '''
    Implements
        dX = -{A @ X}dt + BdW
    or
        dX = -{A @ X}dt + bdW
    where
    -> A is a square drift matrix
    -> B is a square disperion matrix
    -> b is a vector of values of diagonal dispersion matrix
    '''

    def __init__(self, drif_mat, disp_mat_or_vec):

        if disp_mat_or_vec.ndim == 1:  # vector case
            nmd = 0
        elif disp_mat_or_vec.ndim == 2:  # matrix case
            nmd = disp_mat_or_vec.shape[1]
        else:
            raise ValueError("Bad dispersion!")

        self.A = drif_mat
        self.B = disp_mat_or_vec

        super().__init__(self.ou_drift, self.ou_dispersion,
                         noise_mixing_dim=nmd)

    def ou_drift(self, t, x):#, dx):
        # dx[:] = self.A @ x  # need to use [:] because du is a local view
        return self.A @ x

    def ou_dispersion(self, t, x, dx):
        # dx[:] = self.B @ np.ones_like(u)  # only works for diagonal matrices
        dx[:] = self.B[..., np.newaxis]
