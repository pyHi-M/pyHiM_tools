#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 


import SimpleITK as sitk
import argparse
import SimpleITK as sitk
import argparse

def read_image(file_path):
    """Read an image from file."""
    return sitk.ReadImage(file_path)

def write_image(image, file_path):
    """Write an image to file."""
    sitk.WriteImage(image, file_path)

def get_transform_parameters(transform):
    """Extracts and returns translation and rotation parameters from the transform."""
    if isinstance(transform, sitk.Euler3DTransform):
        translation = transform.GetTranslation()
        rotation = transform.GetParameters()[:3]
        return translation, rotation
    else:
        return None, None

def align_images(fixed_image, moving_image):
    """Align the moving_image to the fixed_image using a Demons transformation and return the transformation details."""
    # Initial alignment using a rigid transformation
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                          moving_image, 
                                                          sitk.Euler3DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # Set up the registration framework for rigid transformation
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

    # Execute the initial registration
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))
    
    # Extract translation and rotation parameters
    translation, rotation = get_transform_parameters(final_transform)

    # Deformable registration using Demons algorithm
    demons_filter = sitk.DemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(200)
    demons_filter.SetStandardDeviations(1.0)

    displacement_field = demons_filter.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                               sitk.Cast(moving_image, sitk.sitkFloat32))
    
    displacement_transform = sitk.DisplacementFieldTransform(displacement_field)
    
    return final_transform, translation, rotation, displacement_transform

def resample_image(moving_image, transform, reference_image):
    """Resample the moving image using the provided transform and reference image."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    
    return resampler.Execute(moving_image)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Align two 3D images using SimpleITK.")
    parser.add_argument('--reference', required=True, help='Path to the reference (fixed) image file.')
    parser.add_argument('--moving', required=True, help='Path to the moving image file.')
    parser.add_argument('--output', required=True, help='Path to the output (aligned) image file.')
    parser.add_argument('--displacement_field', required=True, help='Path to save the displacement field image file.')

    args = parser.parse_args()
    print(f"$ Reading images: \n Reference: {args.reference}\n Moving: {args.moving}")

    # Read the images
    fixed_image = read_image(args.reference)
    moving_image = read_image(args.moving)
    
    # Align the images
    print("$ Aligning images in 3D...")

    initial_transform, translation, rotation, displacement_field = align_images(fixed_image, moving_image)
    
    # Apply the initial rigid transform
    moving_image_resampled = resample_image(moving_image, initial_transform, fixed_image)

    # Apply the deformation field using displacement filter
    print("$ Resampling ...")
    displacement_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_filter.SetReferenceImage(fixed_image)
    displacement_filter.SetTransform(sitk.DisplacementFieldTransform(displacement_field))
    
    deformed_image = displacement_filter.Execute(moving_image_resampled)
    
    # Save the result
    write_image(deformed_image, args.output)
    write_image(displacement_field, args.displacement_field)
    print(f"Aligned image saved to {args.output}")
    print(f"Displacement field saved to {args.displacement_field}")
    print("Initial Transform Parameters:")
    print(f"Translation (XYZ): {translation}")
    print(f"Rotation (radians, XYZ): {rotation}")

if __name__ == "__main__":
    main()
