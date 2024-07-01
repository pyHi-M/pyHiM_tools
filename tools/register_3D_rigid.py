#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 

'''
installations

conda create -y -n sitk_env python==3.11
pip install SimpleITK numpy psutil

marcnol, july 2024

'''

import SimpleITK as sitk
import argparse
import numpy as np
import json

def read_image(file_path):
    """Read an image from file."""
    return sitk.ReadImage(file_path)

def write_image(image, file_path):
    """Write an image to file."""
    sitk.WriteImage(image, file_path)


def get_euler_matrix(transform):
    """Get the Euler matrix from a SimpleITK Euler3DTransform."""
    # Get the parameters (rotation and translation)
    parameters = transform.GetParameters()
    # Convert to radians
    angle_x, angle_y, angle_z = np.deg2rad(parameters[:3])
    
    # Compute the rotation matrices for each axis
    rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    
    ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    
    rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    
    # Combine the rotation matrices
    euler_matrix = rz @ ry @ rx
    return euler_matrix

def align_images(fixed_image, moving_image):
    """Align the moving_image to the fixed_image using a rigid transformation."""
    # Initial alignment of the centers of the two images
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                        moving_image, 
                                                        sitk.Euler3DTransform(), 
                                                        sitk.CenteredTransformInitializerFilter.GEOMETRY)

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
    
    # Extract the Euler3DTransform from the CompositeTransform
    euler_transform = None
    for i in range(final_transform.GetNumberOfTransforms()):
        if isinstance(final_transform.GetNthTransform(i), sitk.Euler3DTransform):
            euler_transform = final_transform.GetNthTransform(i)
            break

    if euler_transform is None:
        raise ValueError("No Euler3DTransform found in the composite transform.")

    # Get the Euler matrix
    euler_matrix = get_euler_matrix(euler_transform)
    
    # Get the metric
    metric_value = registration_method.GetMetricValue()
    
    # Get the translation parameters and convert to pyHiM standard shift values
    translation = list(euler_transform.GetTranslation())
    translation[0], translation[1], translation[2] = -translation[1], -translation[0], -translation[2]
    
    transform_details = {
        'shifts_xyz': translation,
        'euler_matrix': euler_matrix,
        'metric_value': metric_value,
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
    parser = argparse.ArgumentParser(description="Align two 3D images using SimpleITK.")
    parser.add_argument('--reference', required=True, help='Path to the reference (fixed) image file.')
    parser.add_argument('--moving', required=True, help='Path to the moving image file.')
    parser.add_argument('--output', required=True, help='Path to the output (aligned) image file.')

    args = parser.parse_args()

    print("This algorithm aligns two 3D volumes using rigid 3D translations an 3D rotations.")

    print(f"$ Reading images: \n Reference: {args.reference}\n Moving: {args.moving}")

    # Read the images
    fixed_image = read_image(args.reference)
    moving_image = read_image(args.moving)
    
    # Align the images
    print("$ Aligning images in 3D...")
    final_transform, transform_details = align_images(fixed_image, moving_image)
        
    # Resample the moving image
    print("$ Resampling ...")
    resampled_moving_image = resample_image(moving_image, final_transform, fixed_image)
    
    # Save the result
    write_image(resampled_moving_image, args.output)
    print(f"Aligned image saved to {args.output}")
    print("Transformation Details:")
    print(f"Shifts (XYZ): {transform_details['shifts_xyz']}")
    print(f"Euler Matrix:\n{transform_details['euler_matrix']}")
    print(f"Metric-value:\n{transform_details['metric_value']}")
    
    output_dict = args.output.split('.')[0]+"_rigid.json"    
    print(f"$ output parameters saved in:\n{output_dict}")
    save_dict_to_json(transform_details, output_dict)

if __name__ == "__main__":
    main()


