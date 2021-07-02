#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 2020

@author: Przemyslaw Zielinski
"""

import numpy as np
from .ito import ItoSDE
from ..reactions import intermediate, Reaction
from itertools import zip_longest
from scipy.special import comb


class ChemicalLangevin(ItoSDE):
    '''
    Implements
        dY = {S @ a(Y)}dt + {S * sqrt(a(Y))}dW
    '''

    def __init__(self, nb_species, ls_reactions):

        self.nb_species = nb_species
        self.nb_reactions = len(ls_reactions)

        self.generate_propensity_parameters(ls_reactions)
        self.generate_stoichiometric_matrix(ls_reactions)

        super().__init__(self.cl_drift, self.cl_dispersion,
                         noise_mixing_dim=self.nb_reactions)

    def cl_drift(self, t, u):
        self.propensity_u = self.propensity(u)
        return self.sm_mat @ self.propensity_u

    def cl_dispersion(self, t, u, du):
        # du[:] = self.sm_mat[..., np.newaxis] * np.sqrt(self.propensity(u))
        du[:] = self.sm_mat[..., np.newaxis] * np.sqrt(self.propensity_u)

    def propensity(self, x):  # x.shape = (nb_species, nb_samples)

        prop = np.ones((self.nb_reactions, x.shape[1]))
        # breakpoint()
        prop = self.ar_rates * prop
        #  we iterate over the number of substrates
        for idcs, coeffs in zip(self.ar_subs_idcs, self.ar_subs_coeffs):
            # breakpoint()
            active = idcs < self.nb_species
            idcs = idcs[active]
            coeffs = coeffs[active][:, np.newaxis]
            prop[active] *= comb(x[idcs], coeffs)

        return prop

    def generate_stoichiometric_matrix(self, ls_reactions):

        sm_mat = []
        for react in ls_reactions:
            sm_vec = np.zeros(self.nb_species)  # stoichiometric vector
            for species_idx, coeff in react.substrates:
                sm_vec[species_idx] = -coeff  # substrates are used up
            for species_idx, coeff in react.products:
                sm_vec[species_idx] = +coeff  # products are created
            sm_mat.append(sm_vec)

        self.sm_mat = np.array(sm_mat).T  # sm_vec's are columns of sm_mat

    def generate_propensity_parameters(self, ls_reactions):
        '''
        Generates three arrays:
            - ar_rates:       with shape (nb_reactions, 1)
                              storing the rates of all reactions
            - ar_subs_idcs:   with shape (max_nb_substrates, nb_reactions)
                              storing the indices of substrates for all reactions
            - ar_subs_coeffs: with shape (max_nb_substrates, nb_reactions)
                              storing the coefficients
        '''

        ls_rates, ar_subs_idcs, ls_subs_coeffs = [], [], []
        for react in ls_reactions:
            ls_rates.append(react.rate)
            ar_subs_idcs.append([subs.species_id for subs in react.substrates])
            ls_subs_coeffs.append([subs.coeff for subs in react.substrates])

        self.ar_rates = np.array(ls_rates)[:, np.newaxis]

        # both 'ar_subs_idcs' and ' ar_subs_coeffs' have shape
        #   (max(nb_substrates), nb_reactions)
        # row s stores sth substrate indices and coefficients for all reactions
        self.ar_subs_idcs  = np.array(list(zip_longest(*ar_subs_idcs,
                                                  fillvalue=self.nb_species)))
        self.ar_subs_coeffs = np.array(list(zip_longest(*ls_subs_coeffs,
                                                   fillvalue=0)))
