#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This version aligns two 3D images using translations only.

'''
installations

conda create -y -n sitk_env python==3.11
pip install SimpleITK numpy psutil

'''

import argparse
import SimpleITK as sitk
import numpy as np
import psutil
import json


def read_image(file_path):
    """Read an image from file."""
    return sitk.ReadImage(file_path)

def write_image(image, file_path):
    """Write an image to file."""
    sitk.WriteImage(image, file_path)

def print_memory_usage(step):
    """Prints the current memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"{step} - Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")

def align_images_translation_only(fixed_image, moving_image):
    """Align the moving_image to the fixed_image using translation only."""
    print_memory_usage("Before alignment")

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
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Execute the registration
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))
    
    print_memory_usage("After alignment")

    # Extract translation parameters
    translation = list(final_transform.GetParameters())
    translation[0], translation[1], translation[2] = -translation[1], -translation[0], -translation[2]

    transform_details = {
        'shifts_xyz': translation,
    }
    
    return final_transform, transform_details

def resample_image(moving_image, transform, reference_image):
    """Resample the moving image using the provided transform and reference image."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    
    return resampler.Execute(moving_image)

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
    parser = argparse.ArgumentParser(description="Align two 3D images using SimpleITK with translation only.")
    parser.add_argument('--reference', required=True, help='Path to the reference (fixed) image file.')
    parser.add_argument('--moving', required=True, help='Path to the moving image file.')
    parser.add_argument('--output', required=True, help='Path to the output (aligned) image file.')

    args = parser.parse_args()

    print("This algorithm aligns two 3D volumes using global registration (translation) in 3D.")

    print(f"$ Reading images: \n Reference: {args.reference}\n Moving: {args.moving}")
    
    # Read the images
    fixed_image = read_image(args.reference)
    moving_image = read_image(args.moving)

    print_memory_usage("Before alignment")
    
    # Align the images
    print("$ Aligning images in 3D...")
    final_transform, transform_details = align_images_translation_only(fixed_image, moving_image)
    
    # Resample the moving image
    print("$ Resampling ...")
    resampled_moving_image = resample_image(moving_image, final_transform, fixed_image)
    
    # Save the result
    write_image(resampled_moving_image, args.output)
    print(f"Aligned image saved to {args.output}")
    print("Translation Parameters (XYZ):", transform_details["shifts_xyz"])

    output_dict = args.output.split('.')[0]+"_global.json"
    print(f"$ output parameters saved in:\n{output_dict}")
    save_dict_to_json(transform_details, output_dict)
    
if __name__ == "__main__":
    main()
