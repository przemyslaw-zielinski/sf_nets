#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 19 Oct 2020

@author: Przemyslaw Zielinski
"""

import os
import torch
import logging
import numpy as np
# import numpy as np
from pathlib import Path
# import sf_nets.utils.dmaps as dmaps
# import spaths
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split

class SimDataset(ABC, Dataset):

    # data files
    train_file = 'train.pt'
    test_file = 'test.pt'
    path_file = 'path.pt'

    def __init__(self, root, train=True, generate=False, transform=None):

        self.root = Path(root)
        self.logger = logging.getLogger(__name__)

        if generate:
            self.logger.info(f"Generating data for {self.name} dataset.")

            (self.root / self.name).mkdir(exist_ok=True)
            self.raw.mkdir(exist_ok=True)
            self.processed.mkdir(exist_ok=True)

            sim_path, train_ds, test_ds = self.generate()

            torch.save(sim_path, self.raw / self.path_file)
            torch.save(train_ds, self.processed / self.train_file)
            torch.save(test_ds, self.processed / self.test_file)

        if not self._check_exists():
            raise RuntimeError('Dataset not found! '
                               'Use generate=True to generate it.')
        if train:
            data_file = self.train_file
        else:
            data_file = self.test_file
        self.load(self.processed / data_file)

    def __repr__(self):
        head = 'Dataset ' + self.name
        body = [f'Number of datapoints: {len(self)}']

        indent = ' ' * 4
        lines = [head] + [indent + line for line in body]

        return '\n'.join(lines)

    @abstractmethod
    def __len__(self):
        """
        Get the length of dataset
        """

    @abstractmethod
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        """

    @abstractmethod
    # @classmethod
    def generate(cls):
        """
        Simulate a raw path as tuple of arrays (times, path)
        and extract train and test datasets.
        """

    @abstractmethod
    def load(self, data_path):
        """
        Create fields to store data and load from file.
        """

    def _check_exists(self):
        # TODO: store and check the simulation metadata
        return (
            (self.processed/self.train_file).exists() and
            (self.processed/self.test_file).exists()
            )

    @property
    def name(self):
        return type(self).__name__

    @property
    def processed(self):
        return self.root / self.name / 'processed'

    @property
    def raw(self):
        return self.root / self.name / 'raw'

def slow_proj(data, sde, solver, nreps, tspan, dt):
    nsam, ndim = data.shape
    ens0 = np.repeat(data, nreps, axis=0)
    nsteps = int(tspan[1]/dt)
    bursts = solver.burst(sde, ens0, (0, nsteps), dt).reshape(nsam, nreps, ndim)
    return np.nanmean(bursts, axis=1)
