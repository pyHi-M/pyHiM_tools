#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply a displacement field to an input 3D image.

Usage:
    python apply_deformation.py --input input_image.tif --displacement displacement_field.tif --output output_image.tif
"""

import argparse
import SimpleITK as sitk

def read_image(file_path):
    """Read an image from file."""
    return sitk.ReadImage(file_path)

def write_image(image, file_path):
    """Write an image to file."""
    sitk.WriteImage(image, file_path)

def apply_displacement_field(image, displacement_field):
    """
    Apply the displacement field to the input image.
    
    Parameters:
    image (SimpleITK.Image): The input image.
    displacement_field (SimpleITK.Image): The displacement field.

    Returns:
    SimpleITK.Image: The deformed image.
    """
    # Cast the displacement field to the required type
    displacement_field = sitk.Cast(displacement_field, sitk.sitkVectorFloat64)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.DisplacementFieldTransform(displacement_field))
    
    deformed_image = resampler.Execute(image)
    return deformed_image

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Apply a displacement field to an input 3D image in TIF format.")
    parser.add_argument('--input', required=True, help='Path to the input image file (TIF format).')
    parser.add_argument('--displacement', required=True, help='Path to the displacement field file (TIF format).')
    parser.add_argument('--output', required=True, help='Path to the output (deformed) image file (TIF format).')

    args = parser.parse_args()

    # Read the input image
    input_image = read_image(args.input)

    # Read the displacement field
    displacement_field = read_image(args.displacement)

    # Apply the displacement field to the input image
    deformed_image = apply_displacement_field(input_image, displacement_field)

    # Save the deformed image
    write_image(deformed_image, args.output)
    print(f"Deformed image saved to {args.output}")

if __name__ == "__main__":
    main()
