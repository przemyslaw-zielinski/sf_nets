#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 13 May 2020

@author: Przemyslaw Zielinski
"""

import numpy as np
import sys, os
sys.path.append(os.path.abspath('../../spaths'))
import spaths

# rqp4
# model params
tsep = 0.04
temp = 0.05
nmd = 4

def rqp4_drif(t, x, dx):
    dx[0] = -x[0] / tsep
    dx[1] = (x[0] - x[1]) / tsep
    dx[2] = (x[1] - x[2]) / tsep
    dx[3] = x[2] - x[3] + x[2]**2 + x[1]**2\
          + 2*x[2]*(x[1]-x[2])/tsep + 2*x[1]*(x[0]-x[1])/tsep + 4*temp/tsep

fast_disp = np.sqrt(2*temp/tsep)
def rqp4_disp(t, x, dx):
    dx[0,0] = fast_disp
    dx[1,1] = fast_disp
    dx[2,2] = fast_disp
    dx[3,1] = 2*x[1]*fast_disp
    dx[3,2] = 2*x[2]*fast_disp
    dx[3,3] = np.sqrt(2*temp)  # = slow_disp

def rqp4_smap(x):
    return x[3] - x[2]**2 - x[1]**2

rqp4_sde = spaths.ItoSDE(rqp4_drif, rqp4_disp, noise_mixing_dim=nmd)
rqp4_dat = {
    'name': 'rqp4',
    'ndim': 4,
    'sdim': 1,
    'nmd': nmd,
    'tsep': tsep,
    'temp': temp,
    'smap': rqp4_smap
}
