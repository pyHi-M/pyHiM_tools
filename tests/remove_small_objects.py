#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:22:32 2023

@author: marcnol
"""

file = "/mnt/grey/DATA/rawData_2023/Experiment_58_Marie_Christophe_Christel_fly_embryo_wt/DAPI_tiff/denoised"+"/scan_001_DAPI_001_ROI_nuclei_denoised1_cp_masks.tif"

from tifffile import imread
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects

def remove_small_masks(im, num_pixels_min =500):
    
    regions = regionprops(im)
    
    n_masks = len(regions)
    
    print(f"$ Number of masks identified: {n_masks}")
    
    print("$ Iterating and removing masks")
    
    im_clean = remove_small_objects(im, min_size = num_pixels_min)
    
    regions_new = regionprops(im_clean)
    n_masks_new = len(regions_new)
    
    print(f"$ Number of masks left: {n_masks_new}")

    return im_clean

im = imread(file)
im_clean = remove_small_masks(im, num_pixels_min =500)