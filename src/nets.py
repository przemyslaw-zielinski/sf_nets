#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 13 May 2020

@author: Przemyslaw Zielinski
"""

import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):

    def __init__(self, net_arch):

        in_features = net_arch['in_features']
        lat_features = net_arch['latent_features']
        # hidden_dims = net_arch.get('hidden_dimensions', [])
        super().__init__()

        encoder = []
        decoder = []

        if hidden_dims := net_arch.get('hidden_dimensions', []):
            for dim in hidden_dims:
                encoder.append(nn.Linear(in_features, dim))
                encoder.append(nn.Sigmoid())
                decoder.insert(0, nn.Linear(dim, in_features))
                decoder.insert(0, nn.Sigmoid())
                in_features = dim
        else:
            print("Constructing network with no hidden layers.")

        # latent view with no activation
        encoder.append(nn.Linear(in_features, lat_features, bias=False))
        decoder.insert(0, nn.Linear(lat_features, in_features, bias=False))

        # register
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):

        z = self.encoder(x)  # latent variable

        return self.decoder(z), z


class MahalanobisLoss(nn.Module):
    def __init__(self, reduction='mean'):  # TODO: implement reduction for batched input
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


class SimpleLabeledDataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
