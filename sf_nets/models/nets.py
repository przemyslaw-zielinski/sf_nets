#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 4 Jun 2020

@author: Przemyslaw Zielinski
"""

import torch.nn as nn
from collections import OrderedDict

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

        n = 0  # in case of no hidden layers
        s = self.coder_size
        if hidden_features:
            for n, dim in enumerate(hidden_features):
                n += 1
                encoder.append((f'layer{n}', nn.Linear(input_features, dim)))
                encoder.append((f'activation{n}', nn.Sigmoid()))
                decoder.insert(0,(f'layer{s-n}', nn.Linear(dim, input_features)))
                decoder.insert(0,(f'activation{s-n-1}', nn.Sigmoid()))
                input_features = dim
        else:
            print("Constructing network with no hidden layers.")

        # latent view with no activation and no bias
        n += 1
        encoder.append((f'layer{n}', nn.Linear(input_features, latent_features,
                                               bias=False)))
        decoder.insert(0,(f'layer{s-n}', nn.Linear(latent_features, input_features,
                                                    bias=False)))

        # register
        self.encoder = nn.Sequential(OrderedDict(encoder))
        self.decoder = nn.Sequential(OrderedDict(decoder))

    @property
    def coder_size(self):  # number of layers (incl. input layer)
        return 2 + len(self.args_dict['hidden_features'])

    @property
    def features(self):
        fs = [self.args_dict['input_features']]
        fs.extend(self.args_dict['hidden_features'])
        fs.append(self.args_dict['latent_features'])
        fs.extend(self.args_dict['hidden_features'][::-1])
        fs.append(self.args_dict['input_features'])
        return fs

    def forward(self, x):

        z = self.encoder(x)  # latent variable

        return self.decoder(z), z
