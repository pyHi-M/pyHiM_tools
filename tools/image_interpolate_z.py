#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script removes every other plane from a 3D image and saves the output in the same format with "_z_reinterpolated" appended to the filename.

Usage:
    python remove_z_planes.py --input input_file

Example:
    python remove_z_planes.py --input image.tif

Arguments:
    --input        Name of the input file (TIFF, NPY, or HDF5 format).

Installation:
    conda create -y -n image_processing python==3.11
    pip install numpy h5py tifffile

Created on Mon Jul  24 2023
Author: marcnol
Updated by OpenAI's GPT-4
"""

import numpy as np
import argparse
import h5py
from tifffile import imread, imwrite
import os

def read_image(file_path):
    """Read an image from a file (TIFF, NPY, or HDF5)."""
    if file_path.endswith('.h5'):
        with h5py.File(file_path, 'r') as f:
            image = f['image'][:]
    elif file_path.endswith('.npy'):
        image = np.load(file_path)
    else:
        image = imread(file_path)
    return image

def write_image(image, file_path):
    """Write an image to a file (TIFF, NPY, or HDF5)."""
    if file_path.endswith('.h5'):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('image', data=image, compression='gzip')
    elif file_path.endswith('.npy'):
        np.save(file_path, image)
    else:
        imwrite(file_path, image)

def _remove_z_planes(image_3d, z_range):
    """
    Removes planes in input image.
    For instance, if you provide a z_range = range(0,image_3d.shape[0],2)

    then the routine will remove any other plane. Number of planes skipped
    can be programmed by tuning z_range.

    Parameters
    ----------
    image_3d : numpy array
        input 3D image.
    z_range : range
        range of planes for the output image.

    Returns
    -------
    output: numpy array
    """
    output = np.zeros((len(z_range), image_3d.shape[1], image_3d.shape[2]))
    for i, index in enumerate(z_range):
        output[i, :, :] = image_3d[index, :, :]

    return output

def parse_arguments():
    parser = argparse.ArgumentParser(description="Remove every other plane from a 3D image.")
    parser.add_argument("--input", required=True, help="Name of the input file (TIFF, NPY, or HDF5 format).")
    return parser.parse_args()

def main():
    args = parse_arguments()
    input_file = args.input

    print(f"Loading file: {input_file}")
    image_3d = read_image(input_file)

    print(f"Original image shape: {image_3d.shape}")
    z_range = range(0, image_3d.shape[0], 2)
    output_image = _remove_z_planes(image_3d, z_range)

    output_file = os.path.splitext(input_file)[0] + '_z_reinterpolated' + os.path.splitext(input_file)[1]
    print(f"Saving output file: {output_file}")
    write_image(output_image, output_file)
    print(f"Output image shape: {output_image.shape}")
    print("Finished execution")

if __name__ == "__main__":
    main()
