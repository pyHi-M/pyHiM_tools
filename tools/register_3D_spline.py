#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 

import SimpleITK as sitk
import argparse

def read_image(file_path):
    """Read an image from file."""
    return sitk.ReadImage(file_path)

def write_image(image, file_path):
    """Write an image to file."""
    sitk.WriteImage(image, file_path)

def align_images(fixed_image, moving_image):
    """Align the moving_image to the fixed_image using a B-spline transformation and return the transformation details."""
    # Initial alignment using a rigid transformation
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

    # Execute the initial registration
    initial_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                    sitk.Cast(moving_image, sitk.sitkFloat32))

    # Deformable registration using B-spline
    grid_physical_spacing = [50.0, 50.0, 50.0]  # control point spacing in physical coordinates
    image_physical_size = [size*spacing for size, spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]
    transform_domain_mesh_size = mesh_size
    bspline_transform = sitk.BSplineTransformInitializer(image1=fixed_image, transformDomainMeshSize=transform_domain_mesh_size, order=3)
    
    # Set the BSplineTransform as the initial transform
    registration_method.SetInitialTransform(bspline_transform, inPlace=False)
    registration_method.SetMetricAsMattesMutualInformation()
    registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0, numberOfIterations=300, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32))

    # Compute the displacement field
    displacement_field = sitk.TransformToDisplacementField(bspline_transform, sitk.sitkVectorFloat64, fixed_image)
    
    return bspline_transform, displacement_field

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

    # Read the images
    fixed_image = read_image(args.reference)
    moving_image = read_image(args.moving)
    
    # Align the images
    final_transform, displacement_field = align_images(fixed_image, moving_image)
    
    # Resample the moving image
    resampled_moving_image = resample_image(moving_image, final_transform, fixed_image)
    
    # Save the result
    write_image(resampled_moving_image, args.output)
    write_image(displacement_field, args.displacement_field)
    print(f"Aligned image saved to {args.output}")
    print(f"Displacement field saved to {args.displacement_field}")

if __name__ == "__main__":
    main()
