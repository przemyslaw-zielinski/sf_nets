#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 4 Jun 2020

@author: Przemyslaw Zielinski
"""

import torch
import torch.nn as nn
from itertools import chain
from collections import OrderedDict

class SimpleAutoencoder(nn.Module):
    '''
    hidden_features
        List containing the details of hidden layers.
        Each element of the list must be in one of the forms:
        -> dim : number of nodes with default activation being Sigmoid
        -> (dim, name) : with 'name' a string containing the name of activation
        -> (dim, name, kwargs) : with 'kwargs' a dict of parameters for 'name'
    '''

    def __init__(self, input_features, latent_features,
                       hidden_features=[]):

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
            for n, feat in enumerate(hidden_features):
                n += 1

                dim, *act_dat = feat if isinstance(feat, list) else (feat,)

                encoder.append((f'layer{n}', nn.Linear(input_features, dim)))
                encoder.append((f'activation{n}', self._init_act(act_dat)))
                decoder.insert(0,(f'layer{s-n}', nn.Linear(dim, input_features)))
                decoder.insert(0,(f'activation{s-n-1}', self._init_act(act_dat)))
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

    @property
    def sparsity(self):
        num = 0.0
        den = 0.0
        for module in chain(self.encoder, self.decoder):
            if hasattr(module, 'weight'):
                num += torch.sum(module.weight == 0)
                den += module.weight.nelement()
            if hasattr(module, 'bias') and module.bias is not None:
                num += torch.sum(module.bias == 0)
                den += module.bias.nelement()
        # for param in self.parameters():  # doesn't see masks!!
        #     num += torch.sum(param == 0)
        #     den += param.nelement()

        return float(num) / float(den)

    def _init_act(self, data):
        if data:
            name, *kwargs = data
            kwargs = (kwargs or [{}])[0]  # if no kwargs default to empty dict
        else:  # default activation
            name = 'Sigmoid'
            kwargs = {}

        return getattr(nn, name)(**kwargs)

    def forward(self, x):

        z = self.encoder(x)  # latent variable

        return self.decoder(z), z
