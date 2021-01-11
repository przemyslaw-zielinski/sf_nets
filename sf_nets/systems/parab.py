"""
Created on Wed 6 Jan 2021

@author: Przemyslaw Zielinski
"""

import numpy as np
import utils.spaths as spaths

# coefficients of hidden process
def hid_drif(lam, eta, gam, eps):
    '''
    All parameters are positive.

    lam : instability strength
    eta : cross-mixing strength between slow variables
    gam : stabilizing coefficient including the fast variable
    eps : time-scale separation
    '''
    def drif(t, x, dx):
        y1, y2, z = x

        dx[0] = lam*y1 - eta*y2 - gam*y1*z
        dx[1] = eta*y1 + lam*y2 - gam*y2*z
        dx[2] = (y1**2 + y2**2 - z) / eps

    return drif

def hid_disp(sig, eps):
    '''
    All parameters are positive.

    sig : noise intensity
    eps : time-scale separation
    '''

    def disp(t, x, dx):
        dx[0] = sig
        dx[1] = sig
        dx[2] = sig / np.sqrt(eps)

    return disp

# tranform to observed coordinates
def transform(x):
    return None

# coefficients of observed process
def obs_drif(lam, eta, gam, eps):

    def drif(t, x, dx):
        pass

    return NotImplementedError()

def obs_disp(sig, eps):

    def disp(t, x, dx):
        pass

    return NotImplementedError()

class ParabSystem(spaths.ItoSDE):

    def __init__(self, lam, eta, gam, sig, eps, hidden=False):

        assert lam > 0.0
        assert eta > 0.0
        assert gam > 0.0
        assert sig > 0.0
        assert eps > 0.0

        self.ndim = 3
        self.nmd = 0
        self.sdim = 2

        if hidden:
            super().__init__(hid_drif(lam, eta, gam, eps),
                             hid_disp(sig, eps),
                             noise_mixing_dim=self.nmd)
        else:
            super().__init__(obs_drif(lam, eta, gam, eps),
                             obs_disp(sig, eps),
                             noise_mixing_dim=self.nmd)

    def eval_lnc(self, data, *args, **kwargs):
        """
        Computes local noise covaraiances for all instances in data.
        """
        return self.ens_diff(0, data)

    def slow_map(self, x):
        y1, y2, z = x
        return y1, y2
