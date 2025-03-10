#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat March 8th 2025
Image Comparison and registration Tool
@author: marcnol


This script allows for:
1. **Comparison of 3D Image Registrations**:
    - Computes similarity metrics (SSIM, MI, RMSE).
    - Generates montages for visual inspection.

2. **Local Shift-Based Image Registration**:
    - Uses user-defined segmented masks to refine 3D alignment.
    - Applies local shifts based on phase correlation.
    - Saves corrected registration images.

Features:
- **Projection Montage:** Creates side-by-side projections of reference vs. target.
- **Mask-Based Registration:** Uses masks for local shift adjustments.
- **Supports Multiple Metrics:** SSIM, MI, RMSE for similarity assessment.
- **Customizable CLI:** Fully configurable via command-line arguments.

Example:
image_register_local_rigid.py --reference scan_005_RT2_043_ROI_ch00_zoom_4.tif 
    --projection_axis 0 
    --target scan_001_mask0_043_ROI_ch00_shifted_zoom_4.tif 
    --mask scan_001_mask0_043_ROI_ch01_3Dmasks_interpolated_zoom_4.tif 
    --do_shift 
    --output montage_corrected_XY

    
Installation:

pip install numpy scipy matplotlib scikit-image tqdm argparse

"""

import os
import sys
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import zoom
from scipy import ndimage
from scipy.ndimage import label
from skimage.measure import regionprops
from skimage.metrics import normalized_mutual_information as mutual_information
from skimage.metrics import normalized_root_mse as root_mse
from skimage.metrics import structural_similarity
from skimage.measure import regionprops, find_contours
from skimage.registration import phase_cross_correlation

def load_image(file_path):
    """
    Load image from file, supporting both .tif and .npy formats
    
    Parameters:
    -----------
    file_path : str
        Path to the image file
    
    Returns:
    --------
    image : ndarray
        Loaded image
    """
    # Get file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # Load based on file extension
        if file_ext == '.npy':
            # Load numpy array
            print(f"Loading numpy array from {file_path}")
            image = np.load(file_path)
        elif file_ext in ['.tif', '.tiff']:
            # Load TIFF image
            print(f"Loading TIFF image from {file_path}")
            image = io.imread(file_path)
        else:
            # Default to skimage.io.imread for other formats
            print(f"Loading image from {file_path} using generic loader")
            image = io.imread(file_path)
        
        # Check if image loaded successfully
        if image is None:
            raise ValueError(f"Failed to load image from {file_path}")
        
        # Print image information
        print(f"  Shape: {image.shape}")
        print(f"  Data type: {image.dtype}")
        print(f"  Min/Max values: {image.min():.4f} / {image.max():.4f}")
        
        return image
    
    except Exception as e:
        print(f"Error loading image from {file_path}: {str(e)}")
        sys.exit(1)

def save_image(image, file_path):
    """
    Save image to file, supporting both .tif and .npy formats
    
    Parameters:
    -----------
    image : ndarray
        Image to save
    file_path : str
        Path to save the image file
    """
    # Get file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Save based on file extension
        if file_ext == '.npy':
            # Save as numpy array
            print(f"Saving numpy array to {file_path}")
            np.save(file_path, image)
        elif file_ext in ['.tif', '.tiff']:
            # Save as TIFF image
            print(f"Saving TIFF image to {file_path}")
            io.imsave(file_path, image)
        else:
            # Default to skimage.io.imsave for other formats
            print(f"Saving image to {file_path} using generic saver")
            io.imsave(file_path, image)
        
        print(f"  Successfully saved image of shape {image.shape} to {file_path}")
        
    except Exception as e:
        print(f"Error saving image to {file_path}: {str(e)}")

def match_segmentation_planes(reference_img, segmented_img):
    """
    Matches the number of planes in the segmented image to the reference image by interpolation if needed.
    
    Parameters:
    -----------
    reference_img : ndarray
        The reference image with the correct number of planes.
    segmented_img : ndarray
        The segmented image to be interpolated if needed.

    Returns:
    --------
    new_segmented_img : ndarray
        The segmented image with the same number of planes as the reference image.
    """
    ref_planes = reference_img.shape[0]
    seg_planes = segmented_img.shape[0]
    
    if seg_planes == ref_planes:
        return segmented_img  # No change needed
    
    scale_factor = ref_planes / seg_planes
    print(f"Interpolating segmented image from {seg_planes} to {ref_planes} planes...")
    
    # Interpolate only along the first dimension
    new_segmented_img = zoom(segmented_img, (scale_factor, 1, 1), order=1)  # Linear interpolation
    
    return new_segmented_img

def relabel_masks(labeled_img, connectivity=3, distance = 3):
    """
    Corrects a labeled mask image by applying watershed segmentation to separate fused masks
    and relabeling connected components to ensure consistency.

    Parameters:
    -----------
    labeled_img : ndarray
        The input labeled image where each object has a unique label.
    connectivity : int, optional
        The connectivity for defining connected components (1=4-connectivity, 2=8-connectivity).
    
    Returns:
    --------
    corrected_labels : ndarray
        The relabeled mask image where touching masks are separated.
    """
    # Count original labels
    original_labels = np.max(labeled_img)

    # rename labels    
    corrected_labels, _ = label(labeled_img, structure=np.ones((3, 3, 3)), output=np.int32)
    
    # Count new labels
    new_labels = np.max(corrected_labels)

    print(f"Original masks: {original_labels}\nNew labels: {new_labels}")
    return corrected_labels


# functions to make montage of reference+target for different masks -


def normalize_image(image):
    """
    Normalizes a 3D floating-point image to the range [0,1] to avoid saturation in RGB plots.

    Parameters:
    -----------
    image : ndarray
        Input 3D floating-point image.

    Returns:
    --------
    normalized_image : ndarray
        The image scaled between 0 and 1.
    """
    image = image.astype(np.float32)  # Ensure float format
    min_val, max_val = np.min(image), np.max(image)

    if max_val > min_val:  # Avoid division by zero
        normalized_image = (image - min_val) / (max_val - min_val)
    else:
        normalized_image = np.zeros_like(image)  # If constant image, return zeros

    return normalized_image
    
def gets_projection(im, axis=0):
    return normalize_image(np.max(im, axis=axis))


def compute_mutual_information(image1, image2):
    """
    Compute the mutual information between two images.

    Parameters:
    -----------
    image1, image2 : ndarray
        Input 3D images to compare.

    Returns:
    --------
    mi : float
        Mutual information value.
    """
    image1, image2 = image1.flatten(), image2.flatten()
    mi = mutual_information(image1, image2)
    return mi

def compute_root_mse(image1, image2):
    """
    Compute the normalized mse between two images.

    Parameters:
    -----------
    image1, image2 : ndarray
        Input 3D images to compare.

    Returns:
    --------
    mse : float
        MSE value
    """
    image1, image2 = image1.flatten(), image2.flatten()
    return root_mse(image1, image2)

def compute_ssm(image1, image2, data_range=65535):
    """
    Compute the normalized mse between two images.

    Parameters:
    -----------
    image1, image2 : ndarray
        Input 3D images to compare.

    Returns:
    --------
    mse : float
        MSE value
    """
    image1, image2 = image1.flatten(), image2.flatten()
    return structural_similarity(image1, image2, data_range = data_range)

def register_3d(
    reference: np.ndarray,
    target: np.ndarray,
    shift_threshold = 5,
    metric: str = "ssim",
    upsample_factor: int = 10,
):
    """
    Perform block-wise 3D image registration

    Parameters:
    -----------
    reference : ndarray
        Reference 3D image block
    target : ndarray
        Target 3D image block to register
    metric : str, optional
        Registration metric ('mi', 'ncc', 'ssim')
    upsample_factor : int, optional
        Subpixel registration precision

    Returns:
    --------
    shift : ndarray
        Estimated shift in ZXY
    error : int
        0: normal termination
        1: did not pass the threshold for shift
    shifted_block : ndarray
        Registered block
    """

    # Normalize for comparison
    ref_norm = (reference - reference.mean()) / (
        reference.std() + 1e-8
    )
    target_norm = (target - target.mean()) / (target.std() + 1e-8)

    data_range = np.max([np.max(ref_norm), np.max(target_norm)])

    # Estimate shifts using phase correlation
    shifts, _, _ = phase_cross_correlation(
        ref_norm, target_norm, upsample_factor=upsample_factor
    )

    if np.max(np.abs(shifts)) < shift_threshold:
        # Apply calculated shifts
        shifted = ndimage.shift(target, shifts, mode="nearest")
        error=0
    else:
        shifted = target
        error=1

    return shifts, error, shifted

def extract_mask_contour(mask):
    """
    Extracts the contour of a binary mask.
    
    Parameters:
    -----------
    mask : ndarray
        2D binary mask image.

    Returns:
    --------
    contour_mask : ndarray
        Image with the mask contour overlaid.
    """
    contour_mask = np.zeros_like(mask)
    contours = find_contours(mask, level=0.5)
    
    for contour in contours:
        contour = np.round(contour).astype(int)
        for y, x in contour:
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                contour_mask[y, x] = 1  # Set contour pixels to 1

    return contour_mask

def plot_montage_projections(reference, target, masks, projection_axis=0, grid_size=4, bbox_size=(30, 30, 30), similarity_thresholds=[0.5, 0.75], do_shift=False, shift_threshold = 5, output='montage.png'):
    """
    Generates a montage gallery of all XY or ZY projections for all masks.
    Each figure contains up to (grid_size x grid_size) masks.

    Parameters:
    -----------
    reference : ndarray
        3D floating point reference image.
    target : ndarray
        3D floating point target image.
    masks : ndarray
        3D int32 labeled mask image.
    projection_axis : int, optional
        Axis along which to compute projections (0 for XY, 1 for ZY).
    grid_size : int, optional
        Defines the number of rows and columns in the montage (default: 4). Must be ≤ 20.

    Returns:
    --------
    None
    """
    # Validate grid size
    if grid_size < 1 or grid_size > 20:
        raise ValueError("grid_size must be between 1 and 20.")

    # Get unique labels in the mask file (excluding background)
    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]  # Remove background (0)
    num_masks = len(unique_labels)

    # Define subplot size dynamically
    fig_width, fig_height = grid_size * 3, grid_size * 3
    masks_per_figure = grid_size ** 2
    num_figures = int(np.ceil(num_masks / masks_per_figure))

    # Process masks in chunks
    for fig_idx in range(num_figures):
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(fig_width, fig_height))
        axes = axes.flatten()  # Flatten axes for easy iteration

        # Select the masks for this figure
        start_idx = fig_idx * masks_per_figure
        end_idx = min(start_idx + masks_per_figure, num_masks)
        selected_labels = unique_labels[start_idx:end_idx]

        for ax, label_id in zip(axes, selected_labels):
            region = regionprops((masks == label_id).astype(np.uint8))[0]  # Get bounding box

            # Extract centroid of the region
            z_c, y_c, x_c = map(int, region.centroid)

            # Define bounding box limits
            z1, z2 = max(0, z_c - bbox_size[0] // 2), min(reference.shape[0], z_c + bbox_size[0] // 2)
            y1, y2 = max(0, y_c - bbox_size[1] // 2), min(reference.shape[1], y_c + bbox_size[1] // 2)
            x1, x2 = max(0, x_c - bbox_size[2] // 2), min(reference.shape[2], x_c + bbox_size[2] // 2)

            # Extract bounding box from images
            ref_crop = normalize_image(reference[z1:z2, y1:y2, x1:x2])
            tgt_crop = normalize_image(target[z1:z2, y1:y2, x1:x2])

            # Extract the mask but **keep only the pixels of the current mask**
            masks_crop = masks[z1:z2, y1:y2, x1:x2].copy()
            masks_crop[masks_crop != label_id] = 0  # Set non-target mask pixels to 0

            # Compute mutual information between reference and target crops
            try:
                similarity = compute_ssm(ref_crop, tgt_crop, data_range=1)
            except ValueError:
                similarity = 0  # Handle any issues in MI calculation

            if do_shift:
                shifts, error, tgt_crop = register_3d(
                    ref_crop,
                    tgt_crop,
                    metric = "ssim",
                    upsample_factor = 10,
                    shift_threshold = shift_threshold,
                )
                if error==0:
                    quality = compute_ssm(ref_crop, tgt_crop, data_range=1)
                else:
                    quality=-1.0 # this flags a shift higher than the threshold
            else:
                shifts=[0.,0.,0.]
                quality=similarity;

            # Compute projections
            ref_proj, tgt_proj, mask_proj = [gets_projection(x, axis=projection_axis) for x in [ref_crop, tgt_crop, masks_crop]]

            # Compute mask contour
            mask_contour = extract_mask_contour(mask_proj)

            # Create RGB projection:
            # - Reference in Magenta (Red + Blue)
            # - Target in Green
            # - Mask as Contour Overlay (White)
            rgb_projection = np.dstack([
                ref_proj,                 # Red Channel (Reference)
                tgt_proj,                 # Green Channel (Target)
                ref_proj.copy()           # Blue Channel (Reference for Magenta Effect)
            ])

            # Overlay mask contour in white (R, G, B)
            rgb_projection[mask_contour == 1] = [1, 1, 1]  # White for mask contour

            # Plot projection
            ax.imshow(rgb_projection, origin='lower')
            
            if quality < similarity_thresholds[0]:
                ax.set_title(f"Label {label_id} | SSM: {quality:.2f}>{similarity:.2f}\nsh=({shifts[0]:.2f},{shifts[1]:.2f},{shifts[2]:.2f})")
            elif quality > similarity_thresholds[0] and quality < similarity_thresholds[1]:
                ax.set_title(f"Label {label_id} | SSM: {quality:.2f}>{similarity:.2f}\nsh=({shifts[0]:.2f},{shifts[1]:.2f},{shifts[2]:.2f})", color='blue')
            elif quality > similarity_thresholds[1]:
                ax.set_title(f"Label {label_id} | SSM: {quality:.2f}>{similarity:.2f}\nsh=({shifts[0]:.2f},{shifts[1]:.2f},{shifts[2]:.2f})", color='green')
            ax.axis('off')

        # Hide any unused subplots
        for i in range(len(selected_labels), len(axes)):
            axes[i].axis('off')

        # Adjust spacing and show figure
        plt.tight_layout()
        
        if projection_axis == 0:
            projection_name = "XY" 
        elif projection_axis == 1:
            projection_name = "ZX" 
        elif projection_axis == 2:
            projection_name = "ZY" 
        plt.suptitle(f"{projection_name} Projection Montage | Page {fig_idx + 1}", fontsize=14)

        # Save **before** plt.show()
        save_path = f"{output.split('.')[0]}_{fig_idx}.{output.split('.')[1]}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

        plt.show()


def parse_arguments():
    """
    Parses command-line arguments for the image comparison and registration tool.

    Returns:
    --------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Image Comparison and Registration Tool"
    )

    # Required arguments
    parser.add_argument(
        "--reference", type=str, required=True, help="Path to the reference 3D image."
    )
    parser.add_argument(
        "--target", type=str, required=True, help="Path to the target 3D image."
    )
    parser.add_argument(
        "--mask", type=str, required=True, help="Path to the segmented mask file."
    )

    # Optional arguments
    parser.add_argument(
        "--projection_axis",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Projection axis: 0 (XY), 1 (ZX), or 2 (ZY) (default: 0).",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        choices=range(1, 21),
        default=10,
        help="Grid size for montage (1-20) (default: 4).",
    )
    parser.add_argument(
        "--bbox_size",
        type=int,
        nargs=3,
        default=(30, 30, 30),
        help="Bounding box size around each mask as three integers (default: 30 30 30).",
    )
    parser.add_argument(
        "--similarity_thresholds",
        type=float,
        nargs=2,
        default=[0.5, 0.75],
        help="Two similarity thresholds for visualization (default: 0.5 0.75).",
    )
    parser.add_argument(
        "--do_shift",
        action="store_true",
        help="Enable local registration correction.",
    )
    parser.add_argument(
        "--shift_threshold",
        type=float,
        default=5,
        help="Threshold for local shifts (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="montage_output.png",
        help="Filename to save the montage image (default: montage_output.png).",
    )

    args = parser.parse_args()

    # Ensure all required files are provided
    if not args.reference or not args.target or not args.mask:
        parser.print_help()
        print("\n⚠️ ERROR: Reference, target, and mask files must be provided.\n")
        exit(1)

    if len(args.output.split('.'))<2:
        args.output = args.output + '.png'
    return args


# Command line interface using argparse

if __name__ == "__main__":
    args = parse_arguments()

    # Load images
    im_ref = load_image(args.reference)
    im_target = load_image(args.target)
    im_segmented = load_image(args.mask)

    # Relabel and process masks
    processed_masks = relabel_masks(im_segmented, connectivity=3, distance=3)

    start_time = time.time()  # Start timer

    # Generate montage
    plot_montage_projections(
        im_ref,
        im_target,
        processed_masks,
        do_shift=args.do_shift,
        projection_axis=args.projection_axis,
        grid_size=args.grid_size,
        bbox_size=tuple(args.bbox_size),
        similarity_thresholds=args.similarity_thresholds,
        shift_threshold=args.shift_threshold,
        output=args.output,
    )
    
    end_time = time.time()  # End timer
    execution_time = end_time - start_time  # Calculate elapsed time

    print(f"Execution Time: {execution_time:.2f} seconds")
