#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This version performs block-wise processing to avoid memory issues.

'''
installations

conda create -y -n deeds python==3.11
pip install git+https://github.com/AlexCoul/deeds-registration@flow_field
pip install SimpleITK numpy psutil

'''

import argparse
import SimpleITK as sitk
import numpy as np
import psutil
from deeds import registration_imwarp_fields

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

def split_image(image_np, factors):
    """Split the 3D image into smaller blocks based on demultiplying factors in XY dimensions."""
    z_size = image_np.shape[0]
    y_size = image_np.shape[1]
    x_size = image_np.shape[2]
    
    block_size = [z_size, y_size // factors[0], x_size // factors[1]]
    blocks = []

    for y in range(factors[0]):
        for x in range(factors[1]):
            y_start = y * block_size[1]
            x_start = x * block_size[2]
            y_end = (y + 1) * block_size[1]
            x_end = (x + 1) * block_size[2]
            
            # Handle the case where the dimensions are not evenly divisible
            if y == factors[0] - 1:
                y_end = y_size
            if x == factors[1] - 1:
                x_end = x_size
                
            block = image_np[:, y_start:y_end, x_start:x_end]
            blocks.append(block)
    return blocks, block_size, factors

def stitch_blocks(blocks, blocks_shape, block_size, original_shape, is_vector=False):
    """Stitch the smaller blocks back into a single 3D image."""
    if is_vector:
        stitched_image = np.zeros(
            (original_shape[0], original_shape[1], original_shape[2], blocks[0].shape[-1]),
            dtype=blocks[0].dtype
        )
    else:
        stitched_image = np.zeros(
            (original_shape[0], original_shape[1], original_shape[2]),
            dtype=blocks[0].dtype
        )

    block_index = 0
    for y in range(blocks_shape[0]):
        for x in range(blocks_shape[1]):
            y_start = y * block_size[1]
            x_start = x * block_size[2]
            y_end = (y + 1) * block_size[1]
            x_end = (x + 1) * block_size[2]
            
            # Handle the case where the dimensions are not evenly divisible
            if y == blocks_shape[0] - 1:
                y_end = original_shape[1]
            if x == blocks_shape[1] - 1:
                x_end = original_shape[2]

            if is_vector:
                stitched_image[:, y_start:y_end, x_start:x_end, :] = blocks[block_index]
            else:
                stitched_image[:, y_start:y_end, x_start:x_end] = blocks[block_index]
            block_index += 1

    return stitched_image

def to_numpy(img):
    return sitk.GetArrayFromImage(img)

def to_sitk(img, ref_img=None):
    img = sitk.GetImageFromArray(img)
    if ref_img:
        img.CopyInformation(ref_img)
    return img

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Align two 3D images using DEEDS with block-wise processing.")
    parser.add_argument('--reference', required=True, help='Path to the reference (fixed) image file.')
    parser.add_argument('--moving', required=True, help='Path to the moving image file.')
    parser.add_argument('--output', required=True, help='Path to the output (aligned) image file.')
    parser.add_argument('--displacement_field', required=True, help='Path to save the displacement field image file.')
    parser.add_argument('--factors', type=int, nargs=2, default=[2, 2], help='Factors to split the image (y, x).')

    args = parser.parse_args()

    # Read the images
    fixed_image = read_image(args.reference)
    moving_image = read_image(args.moving)
    
    fixed_image_np = to_numpy(fixed_image)
    moving_image_np = to_numpy(moving_image)
    
    print(f"Reading images: \n Reference: {args.reference}:{fixed_image_np.shape}\n Moving: {args.moving}:{moving_image_np.shape}")

    print_memory_usage("Before splitting")

    # Split images into blocks
    fixed_blocks, block_size, factors = split_image(fixed_image_np, args.factors)
    moving_blocks, _, _ = split_image(moving_image_np, args.factors)

    print(f"Number of blocks: {len(fixed_blocks)}")
    print_memory_usage("After splitting")

    registered_blocks = []
    displacement_fields_vz = []
    displacement_fields_vy = []
    displacement_fields_vx = []

    # Process each block
    for i, (fixed_block, moving_block) in enumerate(zip(fixed_blocks, moving_blocks)):
        print(f"Processing block {i + 1}/{len(fixed_blocks)} of size {fixed_block.shape}")
        
        # Perform registration
        moved, vz, vy, vx = registration_imwarp_fields(fixed_block, moving_block)
        registered_blocks.append(moved)
        
        # Store displacement fields
        displacement_fields_vz.append(vz)
        displacement_fields_vy.append(vy)
        displacement_fields_vx.append(vx)
        
        print_memory_usage(f"After processing block {i + 1}")

    # Stitch the registered blocks back together
    registered_image_np = stitch_blocks(registered_blocks, factors, block_size, fixed_image_np.shape)
    vz_stitched = stitch_blocks(displacement_fields_vz, factors, block_size, fixed_image_np.shape)
    vy_stitched = stitch_blocks(displacement_fields_vy, factors, block_size, fixed_image_np.shape)
    vx_stitched = stitch_blocks(displacement_fields_vx, factors, block_size, fixed_image_np.shape)

    # Combine the stitched displacement fields
    displacement_fields_np = np.stack([vz_stitched, vy_stitched, vx_stitched], axis=-1)

    registered_image_sitk = to_sitk(registered_image_np, ref_img=fixed_image)
    displacement_field_sitk = sitk.GetImageFromArray(displacement_fields_np, isVector=True)
    displacement_field_sitk.CopyInformation(fixed_image)

    # Save the registered image and displacement field
    write_image(registered_image_sitk, args.output)
    write_image(displacement_field_sitk, args.displacement_field)
    print(f"Aligned image written to {args.output}")
    print(f"Displacement field written to {args.displacement_field}")

    print('done')

if __name__ == "__main__":
    main()
