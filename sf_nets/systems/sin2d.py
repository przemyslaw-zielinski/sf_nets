"""
Created on Tue 17 Nov 2020

@author: Przemyslaw Zielinski
"""

import numpy as np
import spaths

# coefficients of hidden process
def hid_drif(eps):

    def drif(t, x, dx):
        y, z = x
        dx[0] = np.sin(y)
        dx[1] = (np.sin(y) - z) / eps

    return drif

def hid_disp(eps):

    def disp(t, x, dx):
        y, z = x
        dx[0,0] = np.sqrt(1 + .5*np.sin(z))
        dx[1,1] = 1.0 / np.sqrt(eps)

    return disp

# tranform to observed coordinates
def transform(x):
    y, z = x
    return np.array([y+np.sin(z), z])

# coefficients of observed process
def obs_drif(eps):

    def drif(t, x, dx):
        y, z = x
        dx[0] = np.sin(z) + np.cos(z)*(np.sin(y-np.sin(z))-z)/eps \
                - np.sin(z)/(2*eps)
        dx[1] = (np.sin(y-np.sin(z))-z) / eps

    return drif


def obs_disp(eps):

    def disp(t, x, dx):
        y, z = x
        dx[0,0] = np.sqrt(1.0 + 0.5*np.sin(z))
        dx[0,1] = np.cos(z)/np.sqrt(eps)
        dx[1,1] = 1.0 / np.sqrt(eps)

    return disp

class Sin2DSystem(spaths.ItoSDE):

    nmd = 2
    ndim = 2
    sdim = 1

    def __init__(self, eps, hidden=False):

        if hidden:
            super().__init__(hid_drif(eps), hid_disp(eps),
                             noise_mixing_dim=self.nmd)
        else:
            super().__init__(obs_drif(eps), obs_disp(eps),
                             noise_mixing_dim=self.nmd)

    def eval_lnc(self, data, *args, **kwargs):
        """
        Computes local noise covaraiances for all instances in data.
        """
        disp_val = self.ens_disp(0, data)
        return np.einsum('bij,bkj->bik', disp_val, disp_val)

    def slow_map(self, x):
        x1, x2 = x
        return np.array([x1 - np.sin(x2)])

    def slow_map_grad(self, x):
        x1, x2 = x
        return np.array([np.ones_like(x1), -np.cos(x2)])
