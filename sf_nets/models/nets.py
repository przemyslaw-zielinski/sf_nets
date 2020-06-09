#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 4 Jun 2020

@author: Przemyslaw Zielinski
"""

import torch.nn as nn

class SimpleAutoencoder(nn.Module):

    def __init__(self, input_features, latent_features, hidden_features=[]):

        super().__init__()

        self.args_dict = {  # for loading
            'input_features': input_features,
            'latent_features': latent_features,
            'hidden_features': hidden_features
        }

        encoder = []
        decoder = []

        if hidden_features:
            for dim in hidden_features:
                encoder.append(nn.Linear(input_features, dim))
                encoder.append(nn.Sigmoid())
                decoder.insert(0, nn.Linear(dim, input_features))
                decoder.insert(0, nn.Sigmoid())
                input_features = dim
        else:
            print("Constructing network with no hidden layers.")

        # latent view with no activation
        encoder.append(nn.Linear(input_features, latent_features, bias=False))
        decoder.insert(0, nn.Linear(latent_features, input_features, bias=False))

        # register
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):

        z = self.encoder(x)  # latent variable

        return self.decoder(z), z
