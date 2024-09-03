#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 10:42:15 2022

Deforms image anisotropically 

@author: marcnol
"""
import argparse, sys, os
import SimpleITK as sitk
import numpy as np

def check_file_existence(filename):
    """Check if the reference and moving image files exist."""
    if not os.path.exists(filename):
        print(f"Error: image file '{filename}' does not exist.")
        sys.exit(1)

def read_image(file_path):
    """Read an image from file."""
    return sitk.ReadImage(file_path)

def write_image(image, file_path):
    """Write an image to file."""
    sitk.WriteImage(image, file_path)

def create_anisotropic_displacement_field(image, deformation_scales):
    """
    Create an anisotropic displacement field for the given image.
    
    Parameters:
    image (SimpleITK.Image): The input image.
    deformation_scales (tuple): The scale of deformation in (z, y, x) directions.

    Returns:
    SimpleITK.Image: The displacement field.
    """
    size = image.GetSize()
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()
    
    # Create a grid of coordinates
    grid_z, grid_y, grid_x = np.meshgrid(np.linspace(0, size[2]-1, size[2]),
                                         np.linspace(0, size[1]-1, size[1]),
                                         np.linspace(0, size[0]-1, size[0]),
                                         indexing='ij')
    
    # Create an anisotropic displacement field
    displacement_field = np.zeros((size[2], size[1], size[0], 3), dtype=np.float64)
    displacement_field[..., 0] = deformation_scales[2] * np.sin(2.0 * np.pi * grid_x / size[0])
    displacement_field[..., 1] = deformation_scales[1] * np.cos(2.0 * np.pi * grid_y / size[1])
    displacement_field[..., 2] = deformation_scales[0] * np.sin(2.0 * np.pi * grid_z / size[2])
    
    # Convert numpy displacement field to SimpleITK displacement field
    displacement_image = sitk.GetImageFromArray(displacement_field, isVector=True)
    displacement_image.SetSpacing(spacing)
    displacement_image.SetDirection(direction)
    displacement_image.SetOrigin(origin)
    
    return displacement_image

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
    parser = argparse.ArgumentParser(description="Apply anisotropic deformation to an input image in TIF format.")
    parser.add_argument('--input', required=True, help='Path to the input image file (TIF format).')
    parser.add_argument('--deformation_scales', type=float, nargs=3, default=[1.0, 1.0, 1.0], help='Scale of deformation in (z, y, x) directions in pixel units.')

    args = parser.parse_args()
    print("This algorithm deforms a 3D image. Deformations can be anisotropic (see arguments).")

    # Read the input image
    check_file_existence(args.input)
    input_image = read_image(args.input)

    # Create anisotropic displacement field
    deformation_scales = tuple(args.deformation_scales)
    displacement_field = create_anisotropic_displacement_field(input_image, deformation_scales)

    # Apply the displacement field to the input image
    deformed_image = apply_displacement_field(input_image, displacement_field)

    # Save the deformed image
    output_TIFF = args.input.split('.tif')[0] + '_deformed' + '.tif'
    write_image(deformed_image, output_TIFF)
    print(f"Deformed image saved to {output_TIFF}")

if __name__ == "__main__":
    main()
