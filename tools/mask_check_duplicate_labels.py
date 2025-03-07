#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from skimage.measure import label, regionprops
from tifffile import imread
import h5py
from collections import Counter

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

def check_duplicate_labels(mask):
    """
    Check if the same label appears in multiple separate masks.

    Parameters:
        mask (numpy.ndarray): Labeled 3D image.

    Returns:
        None. Prints statistics about duplicated labels.
    """
    print("> Checking for duplicated labels across different masks...")

    # Label the connected components in the mask (to differentiate separate objects)
    labeled_mask = label(mask > 0, connectivity=1)  # Binary connected component labeling

    # Get unique regions and their assigned labels
    regions = regionprops(labeled_mask, intensity_image=mask)

    # Dictionary to track which labels are assigned to different regions
    label_counts = Counter()

    for region in regions:
        region_labels = mask[tuple(region.coords.T)]
        unique_labels = np.unique(region_labels)

        for lbl in unique_labels:
            if lbl > 0:  # Ignore background (zero label)
                label_counts[lbl] += 1

    # Identify duplicated labels
    duplicated_labels = {lbl: count for lbl, count in label_counts.items() if count > 1}

    # Print results
    print(f"Total unique labels: {len(label_counts)}")
    print(f"Labels that appear in multiple masks: {len(duplicated_labels)}")

    if duplicated_labels:
        print("\nDetails of duplicated labels (limited to first 10 for readability):")
        for lbl, count in list(duplicated_labels.items())[:10]:
            print(f"  - Label {lbl} appears in {count} separate masks.")

def main():
    parser = argparse.ArgumentParser(description="Check if any labels are assigned to multiple distinct masks in a labeled image.")
    parser.add_argument("--input", required=True, help="Path to the input labeled image file (TIFF, NPY, or HDF5).")

    args = parser.parse_args()

    # Load the labeled image
    mask = read_image(args.input)

    # Check for duplicated labels
    check_duplicate_labels(mask)

if __name__ == "__main__":
    main()
