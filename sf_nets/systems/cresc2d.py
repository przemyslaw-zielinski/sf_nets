"""
Created on Tue 17 Nov 2020

@author: Przemyslaw Zielinski
"""

import numpy as np
import spaths

# coefficients of hidden process
def hid_drif(a1, a2, a3, a4):

    def drif(t, w, dw):
        u, v = w
        dw[0] = a1
        dw[1] = a3*(1-v)

    return drif

def hid_disp(a1, a2, a3, a4):

    def disp(t, w, dw):
        u, v = w
        dw[0] = a2
        dw[1] = a4

    return disp

# tranform to observed coordinates
def transform(w):
    u, v = w
    return np.array( [v*np.cos(u+v-1), v*np.sin(u+v-1)] )

# coefficients of observed process
def obs_drif(a1, a2, a3, a4):

    def drif(t, x, dx):
        y, z = x
        r = np.sqrt(y**2 + z**2)

        dx[0] = -a1*z + a3*(y/r-z)*(1-r) - (a2**2+a4**2)*y/2 - a2*a4*(z/r+y)\
                - a4**2*z/r
        dx[1] = +a1*y + a3*(z/r+y)*(1-r) - (a2**2+a4**2)*z/2 + a2*a4*(y/r-z)

    return drif


def obs_disp(a1, a2, a3, a4):

    def disp(t, x, dx):
        y, z = x
        r = np.sqrt(y**2 + z**2)

        dx[0,0] = -a2*z
        dx[0,1] = a4*(y/r-z)
        dx[1,0] = a2*y
        dx[1,1] = a4*(z/r+y)

    return disp

class Cresc2DSystem(spaths.ItoSDE):

    ndim = 2
    sdim = 1

    def __init__(self, a1, a2, a3, a4, hidden=False):

        self.nmd = 0 if hidden else self.ndim

        if hidden:
            super().__init__(hid_drif(a1, a2, a3, a4),
                             hid_disp(a1, a2, a3, a4),
                             noise_mixing_dim=1)
        else:
            super().__init__(obs_drif(a1, a2, a3, a4),
                             obs_disp(a1, a2, a3, a4),
                             noise_mixing_dim=self.nmd)

    def eval_lnc(self, data, *args, **kwargs):
        """
        Computes local noise covaraiances for all instances in data.
        """
        return self.ens_diff(0, data)

    def slow_map(self, x):
        y, z = x
        r = np.sqrt(y**2 + z**2)
        return np.array([np.arctan2(z,y) + 1 - r])
