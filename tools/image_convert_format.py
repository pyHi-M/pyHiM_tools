#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert between 3D TIF, HDF5, and NIfTI formats using SimpleITK and h5py.

Usage:
    python convert_image.py --input input_image.tif --output output_image.h5
    python convert_image.py --input input_image.h5 --output output_image.nii.gz
    python convert_image.py --input input_image.nii.gz --output output_image.tif
"""

import argparse
import SimpleITK as sitk
import h5py
import numpy as np
import os

def read_image(file_path):
    """Read a SimpleITK image from a file."""
    return sitk.ReadImage(file_path)

def write_image(image, file_path):
    """Write a SimpleITK image to a file."""
    sitk.WriteImage(image, file_path)

def read_hdf5(file_path):
    """Read a SimpleITK image from an HDF5 file."""
    with h5py.File(file_path, 'r') as h5_file:
        image_np = h5_file['image'][:]
        spacing = tuple(h5_file.attrs['spacing'])
        origin = tuple(h5_file.attrs['origin'])
        direction = tuple(h5_file.attrs['direction'])
    
    image_sitk = sitk.GetImageFromArray(image_np)
    image_sitk.SetSpacing(spacing)
    image_sitk.SetOrigin(origin)
    image_sitk.SetDirection(direction)
    
    return image_sitk

def write_hdf5(image, file_path):
    """Write a SimpleITK image to an HDF5 file."""
    # Convert SimpleITK image to numpy array
    image_np = sitk.GetArrayFromImage(image)

    # Save numpy array to HDF5 file
    with h5py.File(file_path, 'w') as h5_file:
        h5_file.create_dataset('image', data=image_np, compression='gzip')
        h5_file.attrs['spacing'] = image.GetSpacing()
        h5_file.attrs['origin'] = image.GetOrigin()
        h5_file.attrs['direction'] = image.GetDirection()
    
    print(f"Converted image saved to {file_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert between 3D TIF, HDF5, and NIfTI formats using SimpleITK and h5py.")
    parser.add_argument('--input', required=True, help='Path to the input image file (TIF, HDF5, or NIfTI format).')
    parser.add_argument('--output', required=True, help='Path to the output image file (TIF, HDF5, or NIfTI format).')

    args = parser.parse_args()

    input_extension = os.path.splitext(args.input)[1].lower()
    output_extension = os.path.splitext(args.output)[1].lower()

    if input_extension in ['.tif', '.tiff']:
        # Read the input TIF image
        input_image = read_image(args.input)
    elif input_extension in ['.h5', '.hdf5']:
        # Read the input HDF5 image
        input_image = read_hdf5(args.input)
    elif input_extension in ['.nii', '.nii.gz']:
        # Read the input NIfTI image
        input_image = read_image(args.input)
    else:
        raise ValueError("Unsupported input file format")

    if output_extension in ['.tif', '.tiff']:
        # Write the image to TIF format
        write_image(input_image, args.output)
    elif output_extension in ['.h5', '.hdf5']:
        # Write the image to HDF5 format
        write_hdf5(input_image, args.output)
    elif output_extension in ['.nii', '.nii.gz']:
        # Write the image to NIfTI format
        write_image(input_image, args.output)
    else:
        raise ValueError("Unsupported output file format")

if __name__ == "__main__":
    main()
