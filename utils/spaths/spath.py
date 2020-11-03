#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 3 Feb 2020

@author: Przemyslaw Zielinski
"""

import numpy

class StochasticPath():

    def __init__(self, tgrid, xgrid):
        '''
        Stores ensemble of trajectories.
        -> tgrid.shape = (ntimes,)
        -> xgrid.shape = (ntimes, nsams, ndims)
        '''
        self.t = tgrid
        self.x = xgrid
        self.p = numpy.swapaxes(self.x, 0, 1)  # shape = (nsams, ntimes, ndims)

    def __str__(self):
        return (
            f"Stochastic path of {self.x.shape[2]} dimensions: "
            f"from time {self.t[0]} to {self.t[-1]} "
            f"based on {self.x.shape[1]} replicas"
            )

    def __call__(self, times):

        times = numpy.asarray(times)
        scalar_time = False
        if times.ndim == 0:
            times = times[numpy.newaxis]  # Makes x 1D
            scalar_time = True
        if numpy.amin(times) < self.t[0] or numpy.amax(times) > self.t[-1]:
            raise ValueError("Time out of bounds.")

        idxs = []
        for t in times:
            diff = self.t - t
            idxs.append((diff>=0).argmax())

        x = self.x[idxs]
        if scalar_time:
            x = numpy.squeeze(x, axis=0)
        return x
