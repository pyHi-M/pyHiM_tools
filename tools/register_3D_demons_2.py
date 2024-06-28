#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# this version performs downsampling 
import SimpleITK as sitk
import argparse
import psutil

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

def print_memory_usage(step):
    """Prints the current memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"{step} - Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")

def align_images(fixed_image, moving_image):
    """Align the moving_image to the fixed_image using a Demons transformation and return the transformation details."""
    print_memory_usage("Before initial transform")

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
    
    print_memory_usage("After initial transform")

    # Extract translation and rotation parameters
    translation, rotation = get_transform_parameters(final_transform)

    # Deformable registration using Demons algorithm
    demons_filter = sitk.DemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(200)
    demons_filter.SetStandardDeviations(1.0)

    displacement_field = demons_filter.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                               sitk.Cast(moving_image, sitk.sitkFloat32))
    
    displacement_transform = sitk.DisplacementFieldTransform(displacement_field)
    
    print_memory_usage("After Demons registration")
    
    return final_transform, translation, rotation, displacement_transform

def resample_image(moving_image, transform, reference_image):
    """Resample the moving image using the provided transform and reference image."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    
    return resampler.Execute(moving_image)

def downsample_image(image, factor):
    """Downsample the image by the given factor."""
    size = [int(dim / factor) for dim in image.GetSize()]
    return sitk.Resample(image, size, sitk.Transform(), sitk.sitkLinear,
                         image.GetOrigin(), [spacing * factor for spacing in image.GetSpacing()],
                         image.GetDirection(), 0, image.GetPixelID())

def convert_to_supported_type(image):
    """Convert image to a supported type for TIFF."""
    pixel_type = image.GetPixelID()
    if pixel_type in [sitk.sitkUInt8, sitk.sitkInt8, sitk.sitkUInt16, sitk.sitkInt16, sitk.sitkFloat32]:
        return image
    else:
        return sitk.Cast(image, sitk.sitkFloat32)

def convert_displacement_field_to_supported_type(displacement_field):
    """Convert each component of the displacement field to a supported type for TIFF."""
    components = []
    for i in range(displacement_field.GetNumberOfComponentsPerPixel()):
        component_image = sitk.VectorIndexSelectionCast(displacement_field, i, sitk.sitkFloat32)
        components.append(component_image)
    return sitk.Compose(components)

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
    print(f"$ Reading images: \n Reference: {args.reference}\n Moving: {args.moving}")

    print_memory_usage("Before downsampling")
    
    # Downsample images to reduce memory usage
    fixed_image_ds = downsample_image(fixed_image, 2)
    moving_image_ds = downsample_image(moving_image, 2)
    
    print_memory_usage("After downsampling")

    # Align the images
    print("$ Aligning images in 3D...")

    final_transform, translation, rotation, displacement_transform = align_images(fixed_image_ds, moving_image_ds)

    # Apply the initial rigid transform
    print("$ Resampling ...")

    moving_image_resampled = resample_image(moving_image, final_transform, fixed_image)

    # Apply the deformation field using displacement transform
    deformed_image = resample_image(moving_image_resampled, displacement_transform, fixed_image)
    
    # Convert images to supported type for TIFF
    deformed_image = convert_to_supported_type(deformed_image)
    displacement_field_image = convert_displacement_field_to_supported_type(displacement_transform.GetDisplacementField())

    # Save the result
    write_image(deformed_image, args.output)
    write_image(displacement_field_image, args.displacement_field)
    print(f"Aligned image saved to {args.output}")
    print(f"Displacement field saved to {args.displacement_field}")
    if translation and rotation:
        print("Initial Transform Parameters:")
        print(f"Translation (XYZ): {translation}")
        print(f"Rotation (radians, XYZ): {rotation}")

if __name__ == "__main__":
    main()
