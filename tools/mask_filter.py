#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script filters a mask file based on user-provided morphological properties and intensity values from the original image.

The script loads an input file with masks and an optional original intensity image, then filters the masks based on:
- Minimum diameter
- Maximum diameter
- Minimum z position
- Maximum z position
- Eccentricity
- Maximum intensity in the original image

Usage:
    python filter_masks.py --input mask_file --output output_file --min_diameter 5 --max_diameter 50 --min_z 10 --max_z 100 --eccentricity 0.8 --min_intensity 1000 --original original_image_file

Example:
    python filter_masks.py --input mask.tif --output mask_filtered.tif --min_diameter 5 --max_diameter 50 --min_z 10 --max_z 100 --eccentricity 0.8 --min_intensity 1000 --original original.tif

Arguments:
    --input              Name of input mask file (TIFF, NPY, or HDF5 format).
    --output             Name of output file (same format as input).
    --replace_mask_file  Overwrite the input mask file with the filtered masks.
    --min_diameter       Minimum diameter of masks to keep.
    --max_diameter       Maximum diameter of masks to keep.
    --min_z              Minimum z position of masks to keep.
    --max_z              Maximum z position of masks to keep.
    --num_pixels         Minimum number of pixels to keep.
    --min_intensity      Maximum intensity of masks to keep in the original image.
    --intensity_image    Name of original intensity image file (TIFF, NPY, or HDF5 format) if min_intensity is specified.

Installation:
    conda create -y -n mask_analysis python==3.11
    pip install numpy scipy h5py tifffile scikit-image matplotlib

Created on Tue Jul  4 13:26:12 2023
Updated by OpenAI's GPT-4

Author: marcnol
"""

from tifffile import imread, imwrite
import numpy as np
from skimage.measure import regionprops, label
import argparse
import h5py
import sys
import shutil
from tqdm import trange

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

def save_image(image, file_path):
    """Save an image to a file (TIFF, NPY, or HDF5)."""
    if file_path.endswith('.h5'):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('image', data=image, compression='gzip')
    elif file_path.endswith('.npy'):
        np.save(file_path, image)
    else:
        imwrite(file_path, image)

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Name of input mask file (TIFF, NPY, or HDF5 format).")
    parser.add_argument("--output", help="Name of output file (same format as input).", default= None)
    parser.add_argument("--replace_mask_file", action="store_true", help="Overwrite the input mask file with the filtered masks.")
    parser.add_argument("--min_diameter", type=float, help="Minimum diameter of masks to keep.")
    parser.add_argument("--max_diameter", type=float, help="Maximum diameter of masks to keep.")
    parser.add_argument("--min_z", type=int, help="Minimum z position of masks to keep.")
    parser.add_argument("--max_z", type=int, help="Maximum z position of masks to keep.")
    parser.add_argument("--num_pixels", type=float, help="Minimum number of pixels in masks to keep.")
    parser.add_argument("--min_intensity", type=float, help="Minimum intensity of masks to keep in the original image.")
    parser.add_argument("--intensity_image", help="Name of original intensity image file (TIFF, NPY, or HDF5 format) if min_intensity is specified.")

    args = parser.parse_args()

    if args.replace_mask_file and args.output:
        parser.error("Cannot use --replace_mask_file and --output together. Choose one.")

    return args

def filter_masks(labeled_image, original_im=None, min_diameter=None, max_diameter=None, min_z=None, max_z=None, num_pixels=None, min_intensity=None):
    # removes bottom and top planes from labeled image
    
    print("$ filtering out bottom and top planes")
    for plane in trange(labeled_image.shape[0]):
        if (min_z is not None and plane < min_z) or (max_z is not None and plane > max_z):
            labeled_image[plane,:,:] = 0
            
            
    print("$ filtering out min/max diameters, num_pixels, min_intensity")            
    mask = np.zeros(labeled_image.shape, dtype=bool)
    regions = regionprops(labeled_image, intensity_image=original_im if original_im is not None else None)

    print(f'$ Will analyze {len(regions)} labels')
    removed=0
    for region in regions:
        if (min_diameter is not None and region.equivalent_diameter < min_diameter) or \
            (max_diameter is not None and region.equivalent_diameter > max_diameter) or \
            (num_pixels is not None and region.area < num_pixels) or \
            (min_intensity is not None and region.max_intensity < min_intensity):
    
            mask[labeled_image == region.label] = True
            removed+=1
            
    # Set the identified regions to zero
    labeled_image[mask] = 0
    print(f'$ Removed {removed} labels')
    
    return labeled_image

def main():
    args = parseArguments()

    if args.replace_mask_file:
        backup_file = args.input.replace('.tif', '_original.tif').replace('.npy', '_original.npy').replace('.h5', '_original.h5')
        print(f"$ Creating a backup of the original mask file: {backup_file}")
        shutil.copyfile(args.input, backup_file)
        save_path = args.input
    else:
        if args.output is not None:
            save_path = args.output
        else:
            print("Please enter an output file or the --replace_mask_file argument")
            return
    
    print(f"$ Loading mask file: {args.input}")
    im = read_image(args.input)

    original_im = None
    if args.min_intensity is not None:
        if not args.intensity_image:
            print("Error: Original image file must be provided if min_intensity is specified.")
            sys.exit(1)
        print(f"$ Loading original intensity image file: {args.intensity_image}")
        original_im = read_image(args.intensity_image)

    print("$ Filtering masks...")
    filtered_image = filter_masks(im, original_im, args.min_diameter, args.max_diameter, args.min_z, args.max_z, args.num_pixels, args.min_intensity)

    print(f"$ Saving filtered masks to: {save_path}")
    save_image(filtered_image.astype(np.uint8), save_path)

    print("Finished execution")

if __name__ == "__main__":
    main()
