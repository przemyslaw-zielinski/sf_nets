#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 13 Apr 2021

@author: Przemyslaw Zielinski
"""

import sys, os
sys.path[0] = os.getcwd()

import torch
from math import pi
from matplotlib import pyplot as plt
from sf_nets.models.nets import CartToPolar
# from sf_nets.models import CoderNet

ctp = CartToPolar()

print("Testing:", ctp)

b = 10
r = 1.0
p = torch.linspace(0, pi, b)

x = torch.column_stack((r*torch.cos(p), r*torch.sin(p)))

print(x.shape)
# plt.scatter(*x.T)
# plt.show()

y = ctp(x)
print(y)
plt.scatter(*y.T)
plt.xlim([0, 2])
plt.show()
