#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse, sys, select


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-I", "--input", help="Input file against which all files should be compared to"
    )
    p = {}

    args = parser.parse_args()

    if args.input:
        p["input"] = args.input
    else:
        sys.exit("Error: provide input file!")

    p["inputs"] = []
    if select.select(
        [
            sys.stdin,
        ],
        [],
        [],
        0.0,
    )[0]:
        p["inputs"] = [line.rstrip("\n") for line in sys.stdin]
    else:
        print("\nNothing in stdin!")
        sys.exit("Error: provide inputs files using piping!")

    return p

p= parse_arguments()

current_folder=os.getenv('PWD')

files = p["inputs"]
common_file = p["input"]

for file in files:
    output = os.path.basename(file).split('.')[0] + '_scatter_plot.png'

    cmd = 'compare_PWD_matrices.py ' + ' --input1 ' + common_file + ' --input2 ' + file + ' --output ' + output

    print(f'File: {file}, cmd={cmd}')

    os.system(cmd)

