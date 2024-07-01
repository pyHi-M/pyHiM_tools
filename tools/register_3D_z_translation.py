#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This version applies a user-provided XY shift and optimizes the Z shift.

'''
installations

conda create -y -n sitk_env python==3.11
pip install SimpleITK numpy psutil

'''

import argparse
import SimpleITK as sitk
import numpy as np
import json

def read_image(file_path):
    """Read an image from file."""
    return sitk.ReadImage(file_path)

def write_image(image, file_path):
    """Write an image to file."""
    sitk.WriteImage(image, file_path)

def apply_xy_shift(image, shift_xy):
    """Apply the user-provided shift in XY."""
    transform = sitk.TranslationTransform(image.GetDimension())
    transform.SetOffset((shift_xy[0], shift_xy[1], 0.0))
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    shifted_image = resampler.Execute(image)
    return shifted_image

def optimize_z_shift(fixed_image, moving_image):
    """Optimize the shift in Z direction to best align the images."""
    # Initial alignment using a translation transform
    initial_transform = sitk.TranslationTransform(fixed_image.GetDimension())

    # Set up the registration framework
    registration_method = sitk.ImageRegistrationMethod()
    
    # Similarity metric settings
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    # Set the interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set the initial transform
    initial_transform.SetOffset((0.0, 0.0, 0.0))  # Start with no Z shift
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Execute the registration
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))
    
    # Extract Z shift parameter
    z_shift = final_transform.GetOffset()[2]

    transform_details = {
        'z_shift': z_shift,
    }

    return final_transform, transform_details


def save_dict_to_json(dictionary, filename):
    """
    Saves a dictionary to a file in JSON format.

    Parameters:
    dictionary (dict): The dictionary to save.
    filename (str): The name of the file to save the dictionary to.
    """
    try:
        with open(filename, 'w') as json_file:
            json.dump(dictionary, json_file, indent=4)
        print(f"Dictionary successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the dictionary to JSON: {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Apply a user-provided XY shift and optimize the Z shift for 3D image alignment.")
    parser.add_argument('--reference', required=True, help='Path to the reference (fixed) image file.')
    parser.add_argument('--moving', required=True, help='Path to the moving image file.')
    parser.add_argument('--output', required=True, help='Path to the output (aligned) image file.')
    parser.add_argument('--xy_shift', type=float, nargs=2, required=True, help='User-provided XY shift.')

    args = parser.parse_args()

    # Read the images
    fixed_image = read_image(args.reference)
    moving_image = read_image(args.moving)

    # Apply user-provided XY shift
    shifted_moving_image = apply_xy_shift(moving_image, args.xy_shift)

    # Optimize Z shift
    final_transform, transform_details = optimize_z_shift(fixed_image, shifted_moving_image)

    # Apply the final transform including the optimized Z shift
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    aligned_moving_image = resampler.Execute(shifted_moving_image)

    # Save the result
    write_image(aligned_moving_image, args.output)
    print(f"Aligned image saved to {args.output}")
    print(f"Optimized Z shift: {transform_details}")

    output_dict = args.output.split('.')[0]+"_z_translated.json"
    print(f"$ output parameters saved in:\n{output_dict}")
    save_dict_to_json(transform_details, output_dict)
    
if __name__ == "__main__":
    main()
