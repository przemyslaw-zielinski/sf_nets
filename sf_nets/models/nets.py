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

class Coder(OrderedDict):
    '''
    Helper class for constructing sequential networks.

    Combines a list of features with a list of activation functions
    in alternating fashion into an OrderedDict with appropriate names of items.
    Ready to use with nn.Sequential container.

    Args:
        features (list): the dimensions of the network
            features[0] - input dimension
            features[1:] - dimensions of network layers
        activations (list): names of activation functions for the layers or None
            len(activations) = len(features) - 1
            if None or empty str, no activation added on the current layer
    '''

    def __init__(self, features, activations):

        super().__init__()

        inp_dim, hid_dims = features[0], features[1:]
        for n, (out_dim, activ) in enumerate(zip(hid_dims, activations), start=1):
            self[f'layer{n}'] = nn.Linear(inp_dim, out_dim)
            if activ:
                self[f'activation{n}'] = getattr(nn, activ)()
            inp_dim = out_dim


class BaseAutoencoder(nn.Module):
    '''
    Fully connected feed-forward autoencoder with variable number and
    activation type of hidden layers. The hidden layers are symmetric
    for encoder and decoder.

    Args:
        inp_features (int): the dimension of input layer
        lat_features (int): the dimension of latent view
        hid_features (list[int]): dimensions of hidden layers of encoder
            the encoder contains the reverse of this list
        lat_activ (string): the type of activation function on latent view
        out_activ (string): the type of activation function on output layer
        hid_activ (string): the type of activation functions on other layers
    '''

    def __init__(self, inp_features, lat_features, hid_features=[],
                       lat_activ=None, out_activ=None, hid_activ='Tanh'):

        super().__init__()

        self.args_dict = {  # for loading
            'inp_features': inp_features,
            'lat_features': lat_features,
            'hid_features': hid_features,
            'lat_activ': lat_activ,
            'out_activ': out_activ,
            'hid_activ': hid_activ
        }

        # TODO: make it possible to pass a list of strs
        hid_activs = [hid_activ] * len(hid_features)

        enc_features = [inp_features] + hid_features + [lat_features]
        enc_activs = hid_activs + [lat_activ]
        encoder = Coder(enc_features, enc_activs)

        dec_features = [lat_features] + hid_features[::-1] + [inp_features]
        dec_activs = hid_activs[::-1] + [out_activ]
        decoder = Coder(dec_features, dec_activs)

        # register
        self.encoder = nn.Sequential(encoder)
        self.decoder = nn.Sequential(decoder)


    def forward(self, x):
        z = self.encoder(x)  # latent variable
        return self.decoder(z)


    @property
    def coder_size(self):  # number of layers (incl. input layer)
        return 2 + len(self.args_dict['hid_features'])

    @property
    def features(self):
        fs = [self.args_dict['inp_features']]
        fs.extend(self.args_dict['hid_features'])
        fs.append(self.args_dict['lat_features'])
        fs.extend(self.args_dict['hid_features'][::-1])
        fs.append(self.args_dict['inp_features'])
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

    # def _init_activ(self, name, module=nn):
    #     return getattr(nn, name)()

    # def _init_act(self, data):
    #     if data:
    #         name, *kwargs = data
    #         kwargs = (kwargs or [{}])[0]  # if no kwargs default to empty dict
    #     else:  # default activation
    #         name = 'Sigmoid'
    #         kwargs = {}
        #
        # return getattr(nn, name)(**kwargs)
