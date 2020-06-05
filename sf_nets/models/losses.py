#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 4 Jun 2020

@author: Przemyslaw Zielinski
"""
import torch
import torch.nn as nn

class MahalanobisLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()

        if reduction == 'sum':
            self.reduce = lambda x: torch.sum(x)
        elif reduction == 'mean':
            self.reduce = lambda x: torch.mean(x)
        elif reduction == 'none':
            self.reduce = lambda x: x
        else:
            raise ValueError("Unknown reduction type!")

    def forward(self, x, y, covi):

        diff = y - x
        qform = torch.einsum('bn,bnm,bm->b', diff, covi, diff)
        return self.reduce(.5 * qform)
