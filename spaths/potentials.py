"""
Created on Wed 18 Mar 2020

@author: Przemyslaw Zielinski
"""

import numpy as np
from .systems import ItoSDE
from .solvers import EulerMaruyama

class PairwisePotential():

    def __init__(self, W, dim=1, box_length=1):
        self.W = W
        self.dim = dim
        self.box_length = box_length

    def __call__(self, t, x):
        npart = len(x) // self.dim
        # pidx = [tuple(range(i, i+self.dim)) for i in range(npart)]

        val = 0.0
        for i1 in range(npart-1):
            for i2 in range(i1+1, npart):
                # breakpoint()
                p1 = self.get_particle(i1, x)
                p2 = self.get_particle(i2, x)
                dist = box_dist(p1, p2, box_length=self.box_length)
                val += self.W[i1][i2](dist)

        return val

    def get_particle(self, i, x):
        return x[self.dim*i:self.dim*(i+1)]

    def grad(self, t, x):  # x.shape = (ndim, nsam)
        # breakpoint()
        npart = len(x) // self.dim

        grad = np.zeros_like(x)
        for i1 in range(npart-1):
            for i2 in range(i1+1, npart):
                p1 = self.get_particle(i1, x)
                p2 = self.get_particle(i2, x)

                dist = box_dist(p1, p2, box_length=self.box_length)
                diff = box_diff(p1, p2, box_length=self.box_length)
                force = self.W[i1][i2].der(dist) * diff / dist
                # breakpoint()

                grad[self.dim*i1:self.dim*(i1+1)] += force
                grad[self.dim*i2:self.dim*(i2+1)] -= force

        return grad

class DSPotential():
    '''
    Double-state potential with two local minima at distances d1 = compact_state
    and d2 = loose_state, separated by a barrier of height = barrier_height.
    '''
    def __init__(self, barrier_height, compact_state, loose_state):
        self.bh = barrier_height
        self.cs = compact_state
        self.he = (loose_state - compact_state) / 2.0  # half elongation

    def __call__(self, dist):
        arg = (dist - self.cs - self.he)/self.he
        return self.bh * (1 - arg**2)**2

    def der(self, dist):
        arg = (dist - self.cs - self.he)/self.he
        inner_der = -2 * arg
        return 2*self.bh * (1 - arg**2) * inner_der

class WCAPotential():
    '''
    Weeks-Chandler-Andersen (WCA) potential is the Lennard-Jones potential
        eps * ((sigma/r)^12 - 2*(sigma/r)^6)
    truncated at a distance sigma, that corresponds to the minimum potential
    energy of LJ potential.
    It is also shifted upward by the amount eps on the energy scale,
    such that both the energy and force are zero at
    or beyond the cutoff distance.
    '''
    def __init__(self, strength, interaction_distance):
        self.s = strength
        self.id = interaction_distance

    def __call__(self, dist):
        if dist <= self.id:
            r = self.id / dist
            return self.s*(r**12 - 2*r**6 + 1)
        else:
            return 0.0

    def der(self, dist):
        if dist <= self.id:
            r = self.id / dist
            return -12*self.s * (r**13 - r**7) / self.id
        else:
            return 0.0

# helper functions

def initialize_particles(nparts, V, inv_temp, dt, nsteps, rng):

    t_relax = dt*nsteps
    max_val = 1e5
    def relax_gradV(t, x):
        return np.minimum(t * V.grad(t, x) / t_relax, max_val)

    def relax_drift(t, x, dx):
        dx[:] = -relax_gradV(t, x)

    def relax_dispersion(t, x, dx):
        dx[:] = np.sqrt(2 / inv_temp)

    relax_sde = ItoSDE(relax_drift, relax_dispersion)
    relax_ems = EulerMaruyama(rng)
    unif_ens0 = rng.uniform(high=V.box_length, size=(1, 2*nparts))
    relax_sol = relax_ems.solve(relax_sde, unif_ens0, (0.0, t_relax), dt)
    # relax_sol = spaths.EMSolver(relax_sde, unif_ens0, (0.0, t_relax), dt, rng)

    # dimer_distance = [
    #     box_dist(x[0,0:2], x[0,2:4], box_length=V.box_length)
    #     for x in relax_sol.x
    # ]
    # plt.plot(relax_sol.t, dimer_distance, alpha=.6)
    # plt.show()

    return relax_sol.x[nsteps]

def box_fold(x, box_length=1):
    return np.remainder(x, box_length)

def box_diff(p1, p2, box_length=1):
    dp = p1 - p2
    dp -= np.rint(dp / box_length) * box_length
    return dp

def box_dist(p1, p2, box_length=1):
    dp = box_diff(p1, p2, box_length)
    return np.linalg.norm(dp)
