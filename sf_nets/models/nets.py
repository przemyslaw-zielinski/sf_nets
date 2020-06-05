#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 4 Jun 2020

@author: Przemyslaw Zielinski
"""

import torch.nn as nn

class SimpleAutoencoder(nn.Module):

    def __init__(self, arch):

        super().__init__()
        inp_features = arch['input_features']
        lat_features = arch['latent_features']

        encoder = []
        decoder = []

        if hidden_dims := arch.get('hidden_dimensions', []):
            for dim in hidden_dims:
                encoder.append(nn.Linear(inp_features, dim))
                encoder.append(nn.Sigmoid())
                decoder.insert(0, nn.Linear(dim, inp_features))
                decoder.insert(0, nn.Sigmoid())
                inp_features = dim
        else:
            print("Constructing network with no hidden layers.")

        # latent view with no activation
        encoder.append(nn.Linear(inp_features, lat_features, bias=False))
        decoder.insert(0, nn.Linear(lat_features, inp_features, bias=False))

        # register
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):

        z = self.encoder(x)  # latent variable

        return self.decoder(z), z
