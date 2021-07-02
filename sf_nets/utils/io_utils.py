#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 8 Feb 2021

@author: Przemyslaw Zielinski
"""

import sys
from pathlib import Path
from collections import namedtuple

root = Path.cwd()
ProjectPath = namedtuple(
'ProjectPath',
'dataroot, data, configs, figs, models, tabs'
)

def io_path(dataset=""):

    path = ProjectPath(
        root / 'data',
        root / 'data' / dataset,
        root / 'configs' / dataset,
        root / 'results' / 'figs' / dataset,
        root / 'results' / 'models' / dataset,
        root / 'results' / 'tabs'
    )
    path.figs.mkdir(exist_ok=True)

    return path

def get_script_name():
    return Path(sys.argv[0]).stem
