#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 7 Apr 2021

@author: Przemyslaw Zielinski
"""
import torch
import torch.nn as nn

def __getattr__(name):

    if name in globals():
        return globals()['name']
    else:
        activ = getattr(nn, name, None)
        if activ is None:
            raise AttributeError()
        return activ

class Snake(nn.Module):

    def __init__(self, a=1):
        super().__init__()
        self.a = a

    def __call__(self, x):
        return x + (1/self.a) * torch.sin(self.a*x)**2
