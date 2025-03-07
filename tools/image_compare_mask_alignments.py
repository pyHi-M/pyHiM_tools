#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 10:42:15 2022
Image Registration and Comparison Tool
@author: marcnol
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import io, measure, transform, filters, feature, util
from skimage.metrics import structural_similarity
from scipy import ndimage
from scipy.ndimage import zoom
import os
import sys
import argparse
import json
import time

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

def create_comparison_from_segmentation(original_img, registered_img, mask_img, output_dir='mask_comparison_results', dpi=150, display=True, verbose=False):
    """
    Create RGB montages comparing original and registered images for regions defined by masks.
    
    Parameters:
    -----------
    original_img : ndarray
        The original (uncorrected) image
    registered_img : ndarray
        The registered (corrected) image
    mask_img : ndarray
        Segmentation mask image with integer labels (0 = background, 1+ = regions)
    output_dir : str
        Directory to save the comparison montages
    dpi : int
        DPI for saved figures
    display : bool
        Whether to display images during processing (set to False for batch processing)
    verbose : bool
        Whether to print detailed information during processing
    
    Returns:
    --------
    metrics_summary : list
        List of dictionaries containing metrics for each region
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure images have same shape
    assert original_img.shape == registered_img.shape, "Images must have the same dimensions"
    assert original_img.shape[:2] == mask_img.shape[:2], "Mask must have same dimensions as images"
    
    # Convert to grayscale if needed (for display and metrics)
    original_gray = original_img.copy()
    registered_gray = registered_img.copy()
    
    # Normalize images to [0, 1] if needed
    if original_gray.max() > 1.0:
        original_gray = original_gray / 255.0
    if registered_gray.max() > 1.0:
        registered_gray = registered_gray / 255.0
    
    # Create figure for overview with all masks
    """
    fig_overview, ax_overview = plt.subplots(figsize=(12, 10))
    ax_overview.imshow(original_gray, cmap='gray')
    ax_overview.set_title("Overview with all segmented regions")
    """
    
    # Find unique labels in mask (excluding background label 0)
    unique_labels = np.unique(mask_img)
    unique_labels = unique_labels[unique_labels != 0]
    
    print(f"Unique labels: {unique_labels}")

    # Create a colored mask overlay for visualization
    #colored_mask = np.zeros((*mask_img.shape[:2], 4))  # RGBA
    #colored_mask = np.zeros((*mask_img.shape[:2], 3))  # RGB
    
    # Process each region
    metrics_summary = []
    
    for i, label in enumerate(unique_labels):
        # Create binary mask for this region
        region_mask = (mask_img == label)
        
        # Add colored region to overview using RGBA
        '''
        color = plt.cm.tab10(i % 10)  # Get RGBA from colormap
        for y, x in zip(*np.where(region_mask)):
            colored_mask[y, x] = color
        '''
        # Get bounding box for the region

        # Ensure mask is 2D by summing or selecting the appropriate slice
        if region_mask.ndim == 3:
            region_mask = np.max(region_mask, axis=0)  # Collapse along the first axis

        y_indices, x_indices = np.where(region_mask)

        if len(y_indices) == 0 or len(x_indices) == 0:
            continue  # Skip empty regions
            
        x1, x2 = np.min(x_indices), np.max(x_indices)
        y1, y2 = np.min(y_indices), np.max(y_indices)
        
        # Add some padding to the bounding box
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(original_gray.shape[1], x2 + padding)
        y2 = min(original_gray.shape[0], y2 + padding)
        
        # Extract regions
        region_orig = original_gray[y1:y2, x1:x2]
        region_reg = registered_gray[y1:y2, x1:x2]
        region_mask_cropped = region_mask[y1:y2, x1:x2]
        
        # Create RGB comparison (R=original, G=registered, B=0)
        comparison = np.zeros((np.abs(y2-y1), np.abs(x2-x1), 3))
        comparison[:, :, 0] = region_orig  # Red channel = original
        comparison[:, :, 1] = region_reg   # Green channel = registered
        
        # Calculate center of mass for labeling
        cy, cx = ndimage.center_of_mass(region_mask)
        
        # Add rectangle to overview plot
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax_overview.add_patch(rect)
        ax_overview.text(x1, y1-5, f"Region {label}", color='white', 
                         backgroundcolor='black', fontsize=9)
        
        # Create individual region comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Original, Registered, RGB overlay
        axes[0, 0].imshow(region_orig, cmap='gray')
        axes[0, 0].set_title(f"Original - Region {label}")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(region_reg, cmap='gray')
        axes[0, 1].set_title(f"Registered - Region {label}")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(comparison)
        axes[0, 2].set_title("RGB Overlay (R=orig, G=reg)")
        axes[0, 2].axis('off')
        
        # Row 2: Mask overlay, Difference image, Cross-correlation map
        # Show mask overlay on original
        masked_display = np.copy(region_orig)
        if len(masked_display.shape) == 2:
            masked_display = np.stack([masked_display, masked_display, masked_display], axis=2)
        
        # Add mask overlay in yellow
        mask_overlay = np.zeros_like(masked_display)
        mask_overlay[region_mask_cropped, 0] = 1.0  # R
        mask_overlay[region_mask_cropped, 1] = 1.0  # G
        
        axes[1, 0].imshow(region_orig, cmap='gray')
        axes[1, 0].imshow(mask_overlay, alpha=0.3)
        axes[1, 0].set_title("Region Mask Overlay")
        axes[1, 0].axis('off')
        
        # Show difference image
        diff = np.abs(region_orig - region_reg)
        axes[1, 1].imshow(diff, cmap='hot')
        max_diff = diff.max()
        axes[1, 1].set_title(f"Difference (Max: {max_diff:.4f})")
        axes[1, 1].axis('off')
        
        # Calculate correlation map
        corr = feature.match_template(region_orig, region_reg, pad_input=True)
        axes[1, 2].imshow(corr, cmap='viridis')
        y_max, x_max = np.unravel_index(np.argmax(corr), corr.shape)
        axes[1, 2].plot(x_max, y_max, 'r+', markersize=10)
        axes[1, 2].set_title(f"Correlation (Peak: {corr.max():.4f})")
        axes[1, 2].axis('off')
        
        # Calculate metrics only within the mask
        masked_orig = region_orig[region_mask_cropped]
        masked_reg = region_reg[region_mask_cropped]
        mask_size = np.sum(region_mask_cropped)
        
        region_metrics = {}
        
        if len(masked_orig) > 0:  # Ensure we have pixels to compare
            # Calculate MSE within mask
            mse = np.mean((masked_orig - masked_reg) ** 2)
            # Calculate normalized cross-correlation
            ncc = np.corrcoef(masked_orig.flatten(), masked_reg.flatten())[0, 1]
            # Calculate SSIM only on the masked region
            # We use the full regions but pass the mask to structural_similarity
            ssim_val = structural_similarity(region_orig, region_reg, 
                                            data_range=1.0, 
                                            gaussian_weights=True,
                                            sigma=1.5,
                                            use_sample_covariance=False,
                                            mask=region_mask_cropped)
            
            # Store metrics
            region_metrics = {
                'label': label,
                'size': mask_size,
                'mse': mse,
                'ncc': ncc,
                'ssim': ssim_val,
                'max_diff': max_diff,
                'peak_corr': corr.max()
            }
            metrics_summary.append(region_metrics)
            
            # Add metrics to the figure title
            fig.suptitle(f"Region {label} - Size: {mask_size} pixels\n"
                        f"MSE: {mse:.4f}, NCC: {ncc:.4f}, SSIM: {ssim_val:.4f}")
        else:
            fig.suptitle(f"Region {label} - No valid pixels in mask")
        
        # Save figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for suptitle
        plt.savefig(os.path.join(output_dir, f"comparison_region_{label}.png"), dpi=dpi)
        if not display:
            plt.close(fig)
        else:
            plt.show()
    
    # Add mask overlay to overview
    #ax_overview.imshow(colored_mask)
    
    # Save overview figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regions_overview.png"), dpi=dpi)
    if not display:
        plt.close(fig_overview)
    else:
        plt.show()
    
    # Create summary figure with metrics
    if metrics_summary:
        create_metrics_summary(metrics_summary, output_dir)
    
    print(f"Saved {len(unique_labels)} region comparisons to {output_dir}")
    return metrics_summary


def create_metrics_summary(metrics_summary, output_dir):
    """
    Create a summary figure with metrics for all regions.
    
    Parameters:
    -----------
    metrics_summary : list of dict
        List of dictionaries containing metrics for each region
    output_dir : str
        Directory to save the summary figure
    """
    # Sort regions by label
    metrics_summary = sorted(metrics_summary, key=lambda x: x['label'])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data for plotting
    labels = [f"Region {m['label']}" for m in metrics_summary]
    mse_values = [m['mse'] for m in metrics_summary]
    ncc_values = [m['ncc'] for m in metrics_summary]
    ssim_values = [m['ssim'] for m in metrics_summary]
    sizes = [m['size'] for m in metrics_summary]
    
    # Create bar plots
    axes[0, 0].bar(labels, mse_values)
    axes[0, 0].set_title('Mean Squared Error (lower is better)')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_xticklabels(labels, rotation=45)
    
    axes[0, 1].bar(labels, ncc_values)
    axes[0, 1].set_title('Normalized Cross-Correlation (higher is better)')
    axes[0, 1].set_ylabel('NCC')
    axes[0, 1].set_xticklabels(labels, rotation=45)
    
    axes[1, 0].bar(labels, ssim_values)
    axes[1, 0].set_title('Structural Similarity (higher is better)')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].set_xticklabels(labels, rotation=45)
    
    # Size vs. quality scatter plot
    scatter = axes[1, 1].scatter(sizes, ssim_values, c=mse_values, cmap='viridis', 
                                alpha=0.7, s=100)
    axes[1, 1].set_title('Region Size vs. SSIM (color=MSE)')
    axes[1, 1].set_xlabel('Region Size (pixels)')
    axes[1, 1].set_ylabel('SSIM')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('MSE')
    
    # Add annotations for each point
    for i, label in enumerate(labels):
        axes[1, 1].annotate(label, (sizes[i], ssim_values[i]),
                            xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_summary.png'), dpi=dpi)
    if not display:
        plt.close(fig)
    else:
        plt.show()


def register_local_regions(original_img, mask_img, method='phase_correlation', search_window=20, verbose=False):
    """
    Perform local registration based on segmented regions.
    
    Parameters:
    -----------
    original_img : ndarray
        The original image to be registered
    mask_img : ndarray
        Segmentation mask image with integer labels (0 = background, 1+ = regions)
    method : str
        Registration method: 'phase_correlation', 'ecc', or 'feature_based'
    search_window : int
        Size of the search window for local registration
    verbose : bool
        Whether to print detailed information during processing
        
    Returns:
    --------
    registered_img : ndarray
        The registered image
    transformations : dict
        Dictionary of transformations applied to each region
    """
    # Create output image (start with a copy of the original)
    registered_img = np.copy(original_img)
    
    # Convert to grayscale if needed (for registration algorithms)
    if len(original_img.shape) == 3 and original_img.shape[2] > 1:
        original_gray = np.mean(original_img, axis=2)
    else:
        original_gray = original_img
        
    # Find unique labels in mask (excluding background label 0)
    unique_labels = np.unique(mask_img)
    unique_labels = unique_labels[unique_labels != 0]
    
    # Dictionary to store transformations
    transformations = {}
    
    # Process each region
    for label in unique_labels:
        # Create binary mask for this region
        region_mask = (mask_img == label)
        
        # Get bounding box for the region with padding
        y_indices, x_indices = np.where(region_mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue  # Skip empty regions
            
        x1, x2 = np.min(x_indices), np.max(x_indices)
        y1, y2 = np.min(y_indices), np.max(y_indices)
        
        # Add padding for search window
        x1_pad = max(0, x1 - search_window)
        y1_pad = max(0, y1 - search_window)
        x2_pad = min(original_img.shape[1], x2 + search_window)
        y2_pad = min(original_img.shape[0], y2 + search_window)
        
        # Extract padded region and mask
        region = original_gray[y1_pad:y2_pad, x1_pad:x2_pad]
        region_mask_padded = region_mask[y1_pad:y2_pad, x1_pad:x2_pad]
        
        # Apply selected registration method
        if method == 'phase_correlation':
            # Phase correlation for translation only
            shift = register_phase_correlation(region, region_mask_padded)
            transformations[label] = {'shift': shift, 'type': 'translation'}
            
            # Apply shift to the region in the registered image
            if len(original_img.shape) == 3:  # Color image
                for c in range(original_img.shape[2]):
                    registered_img[y1:y2, x1:x2, c] = apply_shift(
                        original_img[y1_pad:y2_pad, x1_pad:x2_pad, c], 
                        shift, region_mask_padded)
            else:  # Grayscale image
                registered_img[y1:y2, x1:x2] = apply_shift(
                    original_img[y1_pad:y2_pad, x1_pad:x2_pad], 
                    shift, region_mask_padded)
        
        elif method == 'ecc':
            # ECC algorithm for affine transformation
            warp_matrix = register_ecc(region, region_mask_padded)
            transformations[label] = {'warp_matrix': warp_matrix.tolist(), 'type': 'affine'}
            
            # Apply warp to the region in the registered image
            if len(original_img.shape) == 3:  # Color image
                for c in range(original_img.shape[2]):
                    registered_img[y1:y2, x1:x2, c] = apply_warp(
                        original_img[y1_pad:y2_pad, x1_pad:x2_pad, c], 
                        warp_matrix, region_mask_padded)
            else:  # Grayscale image
                registered_img[y1:y2, x1:x2] = apply_warp(
                    original_img[y1_pad:y2_pad, x1_pad:x2_pad], 
                    warp_matrix, region_mask_padded)
        
        elif method == 'feature_based':
            # Feature-based registration
            transform_matrix = register_feature_based(region, region_mask_padded)
            transformations[label] = {'transform_matrix': transform_matrix.tolist(), 'type': 'similarity'}
            
            # Apply transform to the region in the registered image
            if len(original_img.shape) == 3:  # Color image
                for c in range(original_img.shape[2]):
                    registered_img[y1:y2, x1:x2, c] = apply_transform(
                        original_img[y1_pad:y2_pad, x1_pad:x2_pad, c], 
                        transform_matrix, region_mask_padded)
            else:  # Grayscale image
                registered_img[y1:y2, x1:x2] = apply_transform(
                    original_img[y1_pad:y2_pad, x1_pad:x2_pad], 
                    transform_matrix, region_mask_padded)
    
    return registered_img, transformations


def register_phase_correlation(image, mask=None):
    """
    Register using phase correlation for translation estimation.
    
    Parameters:
    -----------
    image : ndarray
        The image to register
    mask : ndarray, optional
        Binary mask for the region of interest
    
    Returns:
    --------
    shift : tuple
        (y_shift, x_shift) translation vector
    """
    # Use the center portion of the image as the template for registration
    h, w = image.shape
    center_h, center_w = h // 2, w // 2
    template_size = min(h, w) // 3
    
    # Define template region
    template_y1 = center_h - template_size // 2
    template_y2 = center_h + template_size // 2
    template_x1 = center_w - template_size // 2
    template_x2 = center_w + template_size // 2
    
    template = image[template_y1:template_y2, template_x1:template_x2]
    
    # Apply mask if provided
    if mask is not None:
        masked_image = image * mask
    else:
        masked_image = image
    
    # Calculate phase correlation
    result = feature.match_template(masked_image, template, pad_input=True)
    
    # Find the peak
    y_peak, x_peak = np.unravel_index(np.argmax(result), result.shape)
    
    # Calculate shift relative to the expected position
    y_shift = y_peak - (template_y1 + template_size // 2)
    x_shift = x_peak - (template_x1 + template_size // 2)
    
    return (y_shift, x_shift)


def register_ecc(image, mask=None, num_iterations=50):
    """
    Register using Enhanced Correlation Coefficient (ECC) algorithm.
    
    This is a placeholder - actual implementation would use OpenCV's ECC algorithm,
    which requires cv2 dependency.
    """
    # Placeholder: Would use cv2.findTransformECC in actual implementation
    # Returns an identity transformation for now
    return np.eye(2, 3)


def register_feature_based(image, mask=None):
    """
    Register using feature-based methods (SIFT/ORB/etc.).
    
    This is a placeholder - actual implementation would detect and match features,
    which requires cv2 dependency.
    """
    # Placeholder: Would use cv2.SIFT/ORB + matching in actual implementation
    # Returns an identity transformation for now
    return np.eye(3)


def apply_shift(image, shift, mask=None):
    """
    Apply translation to an image.
    
    Parameters:
    -----------
    image : ndarray
        The image to transform
    shift : tuple
        (y_shift, x_shift) translation vector
    mask : ndarray, optional
        Binary mask for the region of interest
    
    Returns:
    --------
    transformed : ndarray
        The transformed image
    """
    y_shift, x_shift = shift
    transformed = ndimage.shift(image, (y_shift, x_shift), order=3, mode='constant', cval=0)
    
    if mask is not None:
        # Only apply the transformation within the mask
        result = np.copy(image)
        result[mask] = transformed[mask]
        return result
    
    return transformed


def apply_warp(image, warp_matrix, mask=None):
    """
    Apply affine warp to an image.
    
    This is a placeholder - actual implementation would use OpenCV's warpAffine,
    which requires cv2 dependency.
    """
    # Placeholder: Would use cv2.warpAffine in actual implementation
    return image


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

def apply_transform(image, transform_matrix, mask=None):
    """
    Apply general transformation to an image.
    
    This is a placeholder - actual implementation would use OpenCV's warpPerspective,
    which requires cv2 dependency.
    """
    # Placeholder: Would use cv2.warpPerspective in actual implementation
    return image


# Command line interface using argparse
if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Image Registration and Comparison Tool')

    # Add main arguments
    parser.add_argument('--reference', required=True, 
                        help='Path to the reference image')
    parser.add_argument('--target', required=True,
                        help='Path to the target image to be registered/compared')
    parser.add_argument('--segmented', required=True,
                        help='Path to the segmented image with labeled masks')
    parser.add_argument('--mode', choices=['compare', 'register'], default='compare',
                        help='Mode of operation: compare or register+compare')
    parser.add_argument('--method', choices=['phase_correlation', 'ecc', 'feature_based'], 
                        default='phase_correlation',
                        help='Registration method (if mode is register)')
    parser.add_argument('--output', default='results',
                        help='Output directory for results')
       
   

    # Add additional arguments
    parser.add_argument('--save-format', choices=['tif', 'npy'], default=None,
                      help='Override the output format for registered images (default: same as input)')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose output')
    parser.add_argument('--search-window', type=int, default=20,
                      help='Size of search window for local registration (default: 20)')
    parser.add_argument('--no-display', action='store_true',
                      help='Do not display images during processing (useful for batch processing)')
    parser.add_argument('--figure-dpi', type=int, default=150,
                      help='DPI for saved figures (default: 150)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load images
    print(f"\nLoading reference image: {args.reference}")
    reference_img = load_image(args.reference)
    
    print(f"\nLoading target image: {args.target}")
    target_img = load_image(args.target)
    
    print(f"\nLoading segmented image: {args.segmented}")
    segmented_img = load_image(args.segmented)
    segmented_img = match_segmentation_planes(reference_img, segmented_img)
    save_image(segmented_img,args.segmented.split('.')[0]+'_interpolated.npy')

    # Create output directory
    output_dir = args.output
    if args.mode == 'register':
        output_dir = os.path.join(args.output, f"registration_{args.method}")
    else:
        output_dir = os.path.join(args.output, "comparison")
    
    # Process based on selected mode
    if args.mode == 'compare':
        print(f"\nComparing reference and target images using segmentation masks...")
        metrics = create_comparison_from_segmentation(
            reference_img, target_img, segmented_img, output_dir, 
            dpi=args.figure_dpi, display=not args.no_display, verbose=args.verbose)
        print(f"Comparison completed. Results saved to {output_dir}")
        
    elif args.mode == 'register':
        print(f"Registering target to reference using {args.method} method...")
        registered_img, transformations = register_local_regions(
            target_img, segmented_img, method=args.method, 
            search_window=args.search_window, verbose=args.verbose)
        
        # Determine output format
        if args.save_format:
            out_ext = f".{args.save_format}"
        else:
            # Use the same format as the target image
            out_ext = os.path.splitext(args.target)[1]
            if not out_ext:
                out_ext = ".tif"  # Default to .tif if no extension
        
        # Save registered image
        registered_path = os.path.join(output_dir, f"registered_image{out_ext}")
        save_image(registered_img, registered_path)
        print(f"Registered image saved to {registered_path}")
        
        # Save transformation parameters
        transform_path = os.path.join(output_dir, "transformations.json")
        with open(transform_path, 'w') as f:
            json.dump(transformations, f, indent=2)
        print(f"Transformation parameters saved to {transform_path}")
        
        # Compare reference with registered result
        print("Comparing reference and registered images...")
        metrics = create_comparison_from_segmentation(
            reference_img, registered_img, segmented_img, output_dir,
            dpi=args.figure_dpi, display=not args.no_display, verbose=args.verbose)
        print(f"Comparison completed. Results saved to {output_dir}")