#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 10:42:15 2022
Image Registration and Comparison Tool
@author: marcnol
"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import structural_similarity
from scipy.ndimage import zoom
from scipy import ndimage
from scipy.ndimage import label
from skimage.measure import regionprops
from skimage.metrics import structural_similarity as ssim  
from skimage.metrics import normalized_mutual_information as mutual_information

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
    # Flatten images and compute MI
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



def plot_projections_around_masks(reference, target, masks, bbox_size=(40, 20, 20), mask_list=None, max_plots=5):
    """
    Iterates over selected masks in the mask file, extracts a bounding box around each mask, 
    and plots XY and ZY projections in RGB from the reference and target images.

    Parameters:
    -----------
    reference : ndarray
        3D floating point reference image.
    target : ndarray
        3D floating point target image.
    masks : ndarray
        3D int32 labeled mask image.
    bbox_size : tuple of ints
        The size (Z, Y, X) of the bounding box around each mask.
    mask_list : list of ints, optional
        List of mask labels to plot. If None, plots up to `max_plots` masks.
    max_plots : int, optional
        Maximum number of masks to plot if `mask_list` is None.

    Returns:
    --------
    None
    """
    #reference, target= normalize_image(reference), normalize_image(target)

    # Get unique labels in the mask file (excluding background)
    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]  # Remove background (0)

    # If no mask list is provided, limit to the first `max_plots` masks
    if mask_list is None:
        mask_list = unique_labels[:max_plots]
    else:
        mask_list = [label for label in mask_list if label in unique_labels]  # Filter valid masks

    # Iterate over selected masks
    for label_id in mask_list:
        region = regionprops((masks == label_id).astype(np.uint8))[0]  # Get bounding box

        # Extract centroid of the region
        z_c, y_c, x_c = map(int, region.centroid)

        # Define bounding box limits
        z1, z2 = max(0, z_c - bbox_size[0] // 2), min(reference.shape[0], z_c + bbox_size[0] // 2)
        y1, y2 = max(0, y_c - bbox_size[1] // 2), min(reference.shape[1], y_c + bbox_size[1] // 2)
        x1, x2 = max(0, x_c - bbox_size[2] // 2), min(reference.shape[2], x_c + bbox_size[2] // 2)

        # Extract bounding box from images
        ref_crop = reference[z1:z2, y1:y2, x1:x2]
        tgt_crop = target[z1:z2, y1:y2, x1:x2]
        masks_crop = masks[z1:z2, y1:y2, x1:x2]
        #masks_crop = masks_crop[masks_crop == label_id]  

        # Compute similarity (SSIM) between reference and target crops
        try:
            similarity1 = ssim(ref_crop, tgt_crop, data_range=1)
            similarity2 = compute_mutual_information(ref_crop, tgt_crop)
        except ValueError:
            similarity = 0  # Handle any issues in SSIM calculation

        # Compute XY and ZY projections (max intensity)
        ref_xy, tgt_xy, mask_xy = [gets_projection(x, axis=0) for x in [ref_crop,tgt_crop,masks_crop]]
        ref_zy, tgt_zy, mask_zy = [gets_projection(x, axis=1) for x in [ref_crop,tgt_crop,masks_crop]]

        # Create RGB projections
        xy_rgb = np.dstack([ref_xy, tgt_xy, mask_xy])
        zy_rgb = np.dstack([ref_zy, tgt_zy, mask_zy])

        # Plot projections
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(xy_rgb, origin='lower')
        axes[0].set_title(f"XY | Label {label_id}@{x_c},{y_c} | MI: {similarity2:.2f} SSIM: {similarity1:.2f}")
        axes[0].axis('off')

        axes[1].imshow(zy_rgb, origin='lower')
        axes[1].set_title(f"ZY | Label {label_id}@{z_c},{y_c} | MI:{similarity2:.2f} SSIM:{similarity1:.2f}")
        axes[1].axis('off')

        plt.show()


def plot_montage_projections(reference, target, masks, projection_axis=0, grid_size=4, bbox_size = (30, 30, 30)):
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
    fig_width = grid_size * 3  # Scale width
    fig_height = grid_size * 3  # Scale height

    # Determine the number of figures needed
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
            ref_crop = reference[z1:z2, y1:y2, x1:x2]
            tgt_crop = target[z1:z2, y1:y2, x1:x2]
            masks_crop = masks[z1:z2, y1:y2, x1:x2]

            # Compute mutual information between reference and target crops
            try:
                #similarity = compute_mutual_information(ref_crop, tgt_crop)
                similarity = compute_root_mse(ref_crop, tgt_crop)
            except ValueError:
                similarity = 0  # Handle any issues in MI calculation

            # Compute projections
            ref_proj, tgt_proj, mask_proj = [gets_projection(x, axis=projection_axis) for x in [ref_crop, tgt_crop, masks_crop]]

            # Create RGB projection
            rgb_projection = np.dstack([ref_proj, tgt_proj, mask_proj])

            # Determine coordinates to display
            coord_x, coord_y = (x_c, y_c) if projection_axis == 0 else (z_c, y_c)

            # Plot projection
            ax.imshow(rgb_projection, origin='lower')
            ax.set_title(f"Label {label_id} | MI: {similarity:.4f}\n({coord_x},{coord_y})")
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
        plt.show()


# load images

path = '/home/marcnol/data/flybrain/raw_images/testDataset_deformed_brains/'
reference = path+ 'scan_005_RT2_043_ROI_ch00_zoom_4.tif' 
target = path+ 'scan_001_mask0_043_ROI_ch00_shifted_zoom_4.tif'
#segmented =  path+ 'scan_001_mask0_043_ROI_ch01_3Dmasks.npy'
#segmented =  path+ 'scan_001_mask0_043_ROI_ch01_3Dmasks_interpolated.npy'
segmented =  path+ 'scan_001_mask0_043_ROI_ch01_3Dmasks_interpolated_zoom_4.tif'

images_to_load = [reference,target,segmented]
im_ref, im_target, im_segmented = [load_image(x) for x in images_to_load]

# reprocess masks

processed_masks = relabel_masks(im_segmented, connectivity=3, distance = 3)
save_image(processed_masks,segmented.split('.')[0]+'_postprocessed.npy')

plot_projections_around_masks(im_ref, im_target, processed_masks, mask_list=[1,101,150,200],bbox_size=(40, 40, 40))

plot_montage_projections(im_ref, im_target, processed_masks, projection_axis=0, grid_size=10, bbox_size = (30, 30, 30))
