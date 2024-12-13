#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# 
import os
import argparse
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Function to perform maximum intensity projection and save results
def process_image(file_path, output_folder):
    # Load the TIF image using SimpleITK
    print(f'Reading im: {file_path}')
    image = sitk.ReadImage(file_path)
    
    # Convert to numpy array
    image_np = sitk.GetArrayFromImage(image)

    # Perform maximum intensity projection along the Z-axis
    mip = np.max(image_np, axis=0)

    # Save the MIP as a new TIF file
    mip_image = sitk.GetImageFromArray(mip.astype(np.float32))
    mip_output_path = os.path.join(output_folder, os.path.basename(file_path).replace('.tif', '_MIP.tif'))
    sitk.WriteImage(mip_image, mip_output_path)

    # Save the MIP as a PNG with a colormap
    png_output_path = mip_output_path.replace('.tif', '.png')
    print(f'Writing im: {png_output_path}')
    save_colormap_png(mip, png_output_path)

    print(f"Processed {file_path} -> {mip_output_path} and {png_output_path}")

# Function to save an image as a PNG with a colormap
def save_colormap_png(image_np, output_path):
    # Normalize the image
    normalized_image = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))

    # Apply a colormap (e.g., jet)
    colormap_image = cm.jet(normalized_image)

    # Save as PNG
    plt.imsave(output_path, colormap_image, format='png')

# Main function to handle arguments and process images
def main():
    parser = argparse.ArgumentParser(description="Perform Z-axis maximum intensity projection on TIF images and save results.")
    parser.add_argument('--image', type=str, help="Path to a single TIF image file.")
    parser.add_argument('--folder', type=str, help="Path to a folder containing TIF images.")
    parser.add_argument('--output', type=str, required=True, help="Path to the output folder.")

    args = parser.parse_args()

    # Ensure output folder exists
    os.makedirs(args.output, exist_ok=True)

    # Process a single image if specified
    if args.image:
        if not args.image.lower().endswith('.tif'):
            print("Error: The specified file is not a TIF image.")
            return
        process_image(args.image, args.output)

    # Process all images in the folder if specified
    if args.folder:
        for filename in os.listdir(args.folder):
            if filename.lower().endswith('.tif'):
                file_path = os.path.join(args.folder, filename)
                process_image(file_path, args.output)

if __name__ == "__main__":
    main()
