#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 4 Jun 2020

@author: Przemyslaw Zielinski
"""

import torch
import torch.nn as nn
from . import activations as activs
from itertools import chain
from collections import OrderedDict

def assemble_fcnet(features, activations, biases=None):
    '''
    Constructs a sequential, fully connected network by combining a list of
    features with a list of activations, in alternating fashion,
    with sequential names of items.

    Args:
        features (list=[input_dimension, remaining_dimensions...]):
            the dimensions of the network layers
        activations (list): activation functions for the layers or None
            len(activations) = len(features) - 1
            if None, no activation added on the current layer

    Returns:
        fcnet : a Sequential module
    '''

    if not biases:
        biases = [True] * len(features)

    fcnet = OrderedDict()
    inp_dim, hid_dims = features[0], features[1:]
    for n, (out_dim, activ, bias) in enumerate(zip(hid_dims, activations, biases)):
        fcnet[f'layer{n+1}'] = nn.Linear(inp_dim, out_dim, bias=bias)
        if activ:
            fcnet[f'activation{n+1}'] = getattr(activs, activ)()
        inp_dim = out_dim

    return nn.Sequential(fcnet)


class CoderEncoder(nn.Module):
    '''
    Fully connected feed-forward neural network that consists of two subparts:
        - encoder
        - decoder
    such that its input is the input of encoder, its output is the output of
    decoder, and the output layer of encoder is the same as the input layer
    of decoder.

    It has variable number and activation type of hidden layers.
    The hidden layers are symmetric for encoder and decoder.

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
                       lat_activ=None, out_activ=None, hid_activ='ELU',
                       inp_bias=True, lat_bias=True):

        super().__init__()
        # TODO: in init args, activations are already torch.nn objects

        self.args_dict = {  # for loading
            'inp_features': inp_features,
            'lat_features': lat_features,
            'hid_features': hid_features,
            'lat_activ': lat_activ,
            'out_activ': out_activ,
            'hid_activ': hid_activ,
            'inp_bias': inp_bias,
            'lat_bias': lat_bias
        }

        # TODO: make it possible to pass a list of strs
        hid_activs = [hid_activ] * len(hid_features)
        # if not hid_bias:
        if ( nb := len(hid_features) - 1 ) > 0:
            hid_bias = [True] * nb
        else:
            hid_bias = []

        enc_features = [inp_features] + hid_features + [lat_features]
        enc_biases = [inp_bias] + hid_bias + [lat_bias]
        enc_activs = hid_activs + [lat_activ]
        self.encoder = assemble_fcnet(enc_features, enc_activs, enc_biases)

        # TODO: implement non-symmetric case
        dec_features = enc_features[::-1]
        dec_activs = hid_activs[::-1] + [out_activ]
        self.decoder = assemble_fcnet(dec_features, dec_activs)

        self.metrics = {}

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

    def update_metrics(self, batch):
        pass

    def compute_metrics(self):
        return {}

    def reset_metrics(self):
        pass

    def load_state_dict(self, dict, mask=True):
        if mask:
            dict = remove_mask(dict)
        super().load_state_dict(dict)


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

def remove_mask(model_dict):
    mask_state_dict = dict(filter(
        lambda elem: elem[0].endswith('_mask'), model_dict.items()
        ))
    orig_state_dict = dict(filter(
        lambda elem: elem[0].endswith('_orig'), model_dict.items()
        ))
    rest = dict(filter(
        lambda elem: elem[0].endswith(('weight', 'bias')), model_dict.items()
        ))
    state_dict = {
        key.replace('_orig',''): val_orig * val_mask
        for (key, val_orig), val_mask in zip(orig_state_dict.items(),
                                             mask_state_dict.values())
    }
    return {**state_dict, **rest}

class CartToPolar(nn.Module):

    def __init__(self):
        super().__init__()

        self.sqrt = torch.sqrt
        self.atan2 = torch.atan2

    def forward(self, x):

        x1, x2 = x.T

        r = self.sqrt(x1*x1 + x2*x2)
        # p = self.atan2(x2, x1)
        p = torch.sgn(x2) * torch.acos(x1 / r)

        return torch.column_stack((r, p))
