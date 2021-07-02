#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 2020

@author: Przemyslaw Zielinski
"""

from typing import List
from collections import namedtuple
from dataclasses import dataclass, field

intermediate = namedtuple("Intermediate", "species_id coeff", defaults=[1])

@dataclass
class Reaction:
    rate:       float
    substrates: List[intermediate]
    products:   List[intermediate] = field(default_factory=list)
    # inv_rate:   float = 0.0

    def __post_init__(self):
        # make sure all substrates are of class intermediate
        for idx, subs in enumerate(self.substrates):
            if not isinstance(subs, intermediate):
                self.substrates[idx] = intermediate(*subs)

        # make sure all products are of class intermediate
        for idx, prod in enumerate(self.products):
            if not isinstance(prod, intermediate):
                self.products[idx] = intermediate(*prod)

    def __repr__(self):
        # repr of all substrates
        left = ' + '.join(
                            f"{subs.coeff}S_{subs.species_id}"
                            for subs in self.substrates
                         )
        if left == "": left = "0"  # production
        
        # repr of all products
        right = ' + '.join(
                             f"{prod.coeff}S_{prod.species_id}"
                             for prod in self.products
                          )
        if right == "": right = "0"  # degradation substion

        rate_info = f"with rate {self.rate}"
        # one-directional vs equilibrium substions
        # if self.inv_rate == 0.0:
        #     arrow = "-->"
        # else:
        #     arrow = "<->"
        #     rate_info = rate_info + f" and inverse {self.inv_rate}"

        return f"{self.__class__.__name__}({left} --> {right}, {rate_info})"
