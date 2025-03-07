#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from skimage.measure import label, regionprops
from tifffile import imread, imwrite
import h5py

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
    """Save an image in the appropriate format based on file extension."""
    if file_path.endswith('.h5'):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('image', data=image, compression='gzip')
    elif file_path.endswith('.npy'):
        np.save(file_path, image)
    else:
        imwrite(file_path, image.astype(np.uint16))

def fix_duplicate_labels(mask):
    """
    Fix duplicate labels in a labeled image so that each mask has a unique integer.

    Parameters:
        mask (numpy.ndarray): Labeled 3D image.

    Returns:
        fixed_mask (numpy.ndarray): New mask with corrected unique labels.
    """
    print("> Fixing duplicated labels...")

    # Generate new labels for each connected component
    fixed_mask = label(mask > 0, connectivity=1)  # Relabel masks uniquely

    # Print statistics
    unique_labels = np.unique(fixed_mask)
    print(f"Total unique labels after fixing: {len(unique_labels) - 1}")  # Exclude background (0)

    return fixed_mask

def main():
    parser = argparse.ArgumentParser(description="Fix duplicate labels in a labeled image.")
    parser.add_argument("--input", required=True, help="Path to the input labeled image file (TIFF, NPY, or HDF5).")
    parser.add_argument("--output", required=True, help="Path to save the fixed labeled image.")

    args = parser.parse_args()

    # Load the labeled image
    mask = read_image(args.input)

    # Fix duplicate labels
    fixed_mask = fix_duplicate_labels(mask)

    # Save the corrected image
    save_image(fixed_mask, args.output)
    print(f"✅ Fixed labeled image saved to: {args.output}")

if __name__ == "__main__":
    main()
