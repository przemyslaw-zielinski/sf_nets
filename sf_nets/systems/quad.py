"""
Created on Thu 10 Dec 2020

@author: Przemyslaw Zielinski
"""

import numpy as np
import utils.spaths as spaths

# coefficients of hidden process
def hid_drif(Ds, eps):
    '''
    Ds -- dimension of slow coordinates
    eps -- timescale separation
    '''
    def drif(t, x, dx):
        y, z = x[:Ds], x[Ds:]

        dx[:Ds] = 1.0
        dx[Ds:] = -z / eps

    return drif

def hid_disp(Ds, eps):

    def disp(t, x, dx):
        dx[:Ds] = 1.0
        dx[Ds:] = 1.0 / np.sqrt(eps)

    return disp

# tranform to observed coordinates
def transform(x, Ds):
    y, z = x[:Ds], x[Ds:]
    return np.array([y+z**2, z])

# coefficients of observed process
def obs_drif(Ds, eps):

    def drif(t, x, dx):
        x1, x2 = x[:Ds], x[Ds:]

        dx[:Ds] = ( 1.0 + eps - 2*x2[:Ds]**2 ) / eps
        dx[Ds:] = -x2 / eps

    return drif

def obs_disp(Ds, eps):

    def disp(t, x, dx):
        x1, x2 = x[:Ds], x[Ds:]
        i, j = np.indices(dx.shape[:-1])

        dx[(j==i)*(i< Ds)] = 1.0
        dx[(j==i)*(i>=Ds)] = 1.0 / np.sqrt(eps)

        dx[(j==i+Ds)*(i<Ds)] = 2*x2[:Ds] / np.sqrt(eps)

    return disp

class QuadSystem(spaths.ItoSDE):

    def __init__(self, Ds, Df, eps, hidden=False):

        assert Df >= Ds

        self.ndim = Ds + Df
        self.nmd = 0 if hidden else self.ndim
        self.sdim = Ds

        if hidden:
            super().__init__(hid_drif(Ds, eps), hid_disp(Ds, eps),
                             noise_mixing_dim=self.nmd)
        else:
            super().__init__(obs_drif(Ds, eps), obs_disp(Ds, eps),
                             noise_mixing_dim=self.nmd)

    def eval_lnc(self, data, *args, **kwargs):
        """
        Computes local noise covaraiances for all instances in data.
        """
        disp_val = self.ens_disp(0, data)
        return np.einsum('bij,bkj->bik', disp_val, disp_val)

    def slow_map(self, x):
        Ds = self.sdim
        x1, x2 = x[:Ds], x[Ds:]
        return x1 - x2[:Ds]**2
