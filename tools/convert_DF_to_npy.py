
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 


import SimpleITK as sitk
import numpy as np

def read_displacement_field(file_path):
    """
    Read a displacement field from a multi-channel (color) image.

    Parameters:
    file_path (str): Path to the displacement field image file.

    Returns:
    tuple: Three numpy arrays representing the x, y, and z versors.
    """
    # Read the multi-channel image
    displacement_image = sitk.ReadImage(file_path)

    # Convert to numpy array
    displacement_array = sitk.GetArrayFromImage(displacement_image)

    # Split the channels into three matrices
    vx = displacement_array[..., 0]
    vy = displacement_array[..., 1]
    vz = displacement_array[..., 2]

    return vx, vy, vz

def main():
    # Path to the displacement field image file

    path = "/home/marcnol/data/blobel/000/"

    file_path = path + 'scan_001_RT301_000_ROI_converted_decon_ch00_DF.tif'

    # Read and decode the displacement field
    vx, vy, vz = read_displacement_field(file_path)

    # Print the shapes of the decoded matrices
    print(f"vx shape: {vx.shape}")
    print(f"vy shape: {vy.shape}")
    print(f"vz shape: {vz.shape}")

    # Optionally, save the decoded matrices to separate files (example)
    np.save('vx.npy', vx)
    np.save('vy.npy', vy)
    np.save('vz.npy', vz)

if __name__ == "__main__":
    main()
