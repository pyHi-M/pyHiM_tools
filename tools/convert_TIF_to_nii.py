
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 

import SimpleITK as sitk
import numpy as np

def save_displacement_field_nifti(vx, vy, vz, reference_image, output_path):
    """
    Save the displacement field as a compressed NIfTI file (.nii.gz).
    
    Parameters:
    vx, vy, vz (numpy.ndarray): Displacement field components.
    reference_image (SimpleITK.Image): Reference image to copy spatial information.
    output_path (str): Path to save the NIfTI file.
    """
    # Stack the displacement field components into a single array
    displacement_field = np.stack((vx, vy, vz), axis=-1)
    
    # Convert to SimpleITK image
    displacement_image = sitk.GetImageFromArray(displacement_field, isVector=True)
    displacement_image.CopyInformation(reference_image)
    
    # Save as NIfTI file with compression
    sitk.WriteImage(displacement_image, output_path, useCompression=True)
    print(f"Displacement field saved to {output_path}")

def main():
    # Example usage
    # Assuming vx, vy, vz are your displacement field components
    # and reference_image is the original image from which these fields were computed
    
    # Replace these with actual data
    vx = np.random.rand(65, 2048, 2048)
    vy = np.random.rand(65, 2048, 2048)
    vz = np.random.rand(65, 2048, 2048)
    path = "/home/marcnol/data/blobel/000/"

    reference_image_path = path + 'scan_001_RT301_000_ROI_converted_decon_ch00_DF.tif'
    output_path = 'scan_001_RT301_000_ROI_converted_decon_ch00_DF.nii.gz'

    reference_image = sitk.ReadImage(reference_image_path)
    
    save_displacement_field_nifti(vx, vy, vz, reference_image, output_path)

if __name__ == "__main__":
    main()
