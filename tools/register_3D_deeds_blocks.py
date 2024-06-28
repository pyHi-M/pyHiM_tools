#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# this version performs downsampling 

'''
installations

conda create -y -n deeds python==3.11
pip install git+https://github.com/AlexCoul/deeds-registration@flow_field
pip install simpleITK numpy psutil

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

def split_image(image_np, block_size):
    """Split the 3D image into smaller blocks."""
    z_blocks = image_np.shape[0] // block_size[0]
    y_blocks = image_np.shape[1] // block_size[1]
    x_blocks = image_np.shape[2] // block_size[2]
    blocks = []

    for z in range(z_blocks):
        for y in range(y_blocks):
            for x in range(x_blocks):
                block = image_np[
                    z * block_size[0]:(z + 1) * block_size[0],
                    y * block_size[1]:(y + 1) * block_size[1],
                    x * block_size[2]:(x + 1) * block_size[2]
                ]
                blocks.append(block)
    return blocks, (z_blocks, y_blocks, x_blocks)

def stitch_blocks(blocks, blocks_shape, block_size):
    """Stitch the smaller blocks back into a single 3D image."""
    stitched_image = np.zeros((
        blocks_shape[0] * block_size[0],
        blocks_shape[1] * block_size[1],
        blocks_shape[2] * block_size[2]
    ), dtype=blocks[0].dtype)

    block_index = 0
    for z in range(blocks_shape[0]):
        for y in range(blocks_shape[1]):
            for x in range(blocks_shape[2]):
                stitched_image[
                    z * block_size[0]:(z + 1) * block_size[0],
                    y * block_size[1]:(y + 1) * block_size[1],
                    x * block_size[2]:(x + 1) * block_size[2]
                ] = blocks[block_index]
                block_index += 1

    return stitched_image

def to_numpy(img):
    result = sitk.GetArrayFromImage(img)
    return result

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
    parser.add_argument('--block_size', type=int, nargs=3, default=[512, 512, 65], help='Block size for splitting the image (z, y, x).')

    args = parser.parse_args()

    # Read the images
    fixed_image = read_image(args.reference)
    moving_image = read_image(args.moving)
    
    fixed_image_np = to_numpy(fixed_image)
    moving_image_np = to_numpy(moving_image)
    
    print(f"Reading images: \n Reference: {args.reference}:{fixed_image_np.shape}\n Moving: {args.moving}:{moving_image_np.shape}")

    print_memory_usage("Before splitting")

    # Split images into blocks
    fixed_blocks, blocks_shape = split_image(fixed_image_np, args.block_size)
    moving_blocks, _ = split_image(moving_image_np, args.block_size)

    print(f"Number of blocks: {len(fixed_blocks)}")
    print_memory_usage("After splitting")

    registered_blocks = []
    displacement_fields = []

    # Process each block
    for i, (fixed_block, moving_block) in enumerate(zip(fixed_blocks, moving_blocks)):
        print(f"Processing block {i + 1}/{len(fixed_blocks)}")
        
        # Perform registration
        moved, vz, vy, vx = registration_imwarp_fields(fixed_block, moving_block)
        registered_blocks.append(moved)
        
        # Store displacement fields
        displacement_field = np.stack([vz, vy, vx], axis=-1).astype(np.float32)
        displacement_fields.append(displacement_field)
        
        print_memory_usage(f"After processing block {i + 1}")

    # Stitch the registered blocks back together
    registered_image_np = stitch_blocks(registered_blocks, blocks_shape, args.block_size)
    displacement_fields_np = np.concatenate(displacement_fields, axis=0)

    registered_image_sitk = to_sitk(registered_image_np, ref_img=fixed_image)
    displacement_field_sitk = sitk.GetImageFromArray(displacement_fields_np, isVector=True)

    # Save the registered image and displacement field
    write_image(registered_image_sitk, args.output)
    write_image(displacement_field_sitk, args.displacement_field)
    print(f"Aligned image written to {args.output}")
    print(f"Displacement field written to {args.displacement_field}")

    print('done')

if __name__ == "__main__":
    main()
