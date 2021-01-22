#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 25 Nov 2020

@author: Przemyslaw Zielinski
"""
import argparse
from pathlib import Path
from subprocess import call

def main(dataset, regex):

    cfg_dir = Path('.') / 'configs' / dataset
    file_list = [
        file_path
        for file_path in cfg_dir.iterdir()
        if regex in file_path.stem
    ]

    print(f"Models selected to train:")
    for file_path in file_list:
        print("\t", file_path.stem)

    proceed = input("Proceed ([y]/n)? ")
    if proceed.lower() == "y" or "":
        for file_path in file_list:
            call(["./train.py", file_path])

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='Batch training')
    args.add_argument('dataset', type=str, help="Select dataset")
    args.add_argument('--regex', type=str, default="",
                      help='Select models satisfying this expression')
    args = args.parse_args()

    # TODO: use real regexes with re library
    main(args.dataset, args.regex)