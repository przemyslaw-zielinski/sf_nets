#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 25 Nov 2020

@author: Przemyslaw Zielinski
"""
import argparse
from pathlib import Path
from subprocess import call

def main(dataset, regex, nreps):

    cfg_dir = Path('.') / 'configs' / dataset
    file_list = [
        file_path
        for file_path in cfg_dir.iterdir()
        if regex in file_path.stem
    ]

    print(f"Models selected to train:")
    for file_path in file_list:
        print("\t", file_path.stem)
    if nreps>0:
        print(f"Number of replicas for each file: {nreps}.")

    proceed = input("Proceed (y/n)? ")
    if proceed.lower() == "y":
        for file_path in file_list:
            for n in range(nreps):
                call(["./train.py", file_path, f"-r {n}"])

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='Batch training')
    args.add_argument('dataset', type=str, help="Select dataset")
    args.add_argument('--regex', type=str, default="",
                      help='Select models satisfying this expression')
    args.add_argument('--nreps', type=int, default=0,
                      help="Number of replicas")
    args = args.parse_args()

    # TODO: use real regexes with re library
    main(args.dataset, args.regex, args.nreps)
