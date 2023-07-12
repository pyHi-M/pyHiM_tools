#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:40:38 2022

@author: marcnol

This script will 2D project the 3D labeled numpy arrays produced by segmentMasks3D and replace those produced by segmentMasks
s
"""

import argparse
import glob
import os
from datetime import datetime
import select
import sys
from tifffile import imread, imsave
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Name of input trace file.")
    parser.add_argument("--num_pixels_min", help="Masks with less that this number of pixels will be removed. Default = 0")
    parser.add_argument("--save", help="Saves the processed 3D image", action="store_true")
    parser.add_argument("--convert", help="Converts from NPY to TIF or vice versa depending on input", action="store_true")
    parser.add_argument("--z_min", help="Z minimum for a localization. Default = 0")
    parser.add_argument("--z_max", help="Z maximum for a localization. Default = np.inf")
    parser.add_argument("--y_min", help="Y minimum for a localization. Default = 0")
    parser.add_argument("--y_max", help="Y maximum for a localization. Default = np.inf")
    parser.add_argument("--x_min", help="X minimum for a localization. Default = 0")
    parser.add_argument("--x_max", help="X maximum for a localization. Default = np.inf")    
    parser.add_argument("--pipe", help="inputs Trace file list from stdin (pipe)", action="store_true")

    p = {}

    args = parser.parse_args()

    if args.input:
        p["input"] = args.input
    else:
        p["input"] = None
      
    if args.z_min:
        p["z_min"] = int(args.z_min)
    else:
        p["z_min"] = 0

    if args.z_max:
        p["z_max"] = int(args.z_max)
    else:
        p["z_max"] = np.inf

    if args.y_min:
        p["y_min"] = float(args.y_min)
    else:
        p["y_min"] = 0

    if args.y_max:
        p["y_max"] = float(args.y_max)
    else:
        p["y_max"] = np.inf

    if args.x_min:
        p["x_min"] = float(args.x_min)
    else:
        p["x_min"] = 0

    if args.x_max:
        p["x_max"] = float(args.x_max)
    else:
        p["x_max"] = np.inf
        
    if args.num_pixels_min:
        p["num_pixels_min"] = int(args.num_pixels_min)
    else:
        p["num_pixels_min"] = 0

    if args.save:
        p["save"] = True
    else:
        p["save"] = False
        
    if args.convert:
        p["convert"] = True
    else:
        p["convert"] = False
        
    p["files"] = []
    if args.pipe:
        p["pipe"] = True
        if select.select([sys.stdin,], [], [], 0.0)[0]:
            p["files"] = [line.rstrip("\n") for line in sys.stdin]
        else:
            print("Nothing in stdin")
    else:
        p["pipe"] = False
        p["files"] = [p["input"]]

    return p


def renames_file(output_file):
    if os.path.exists(output_file):
            print(f"----Warning!----\nRenamed {output_file} as it exists already!\n")
            os.rename(output_file, output_file.split('.')[0] + "_old" + '.' + output_file.split('.')[1])
        
def save_projections(file, im_2d):
    output_file = file.split(".")[0] + "_Masks.npy" 
    output_file_TIF = file.split(".")[0] + "_Masks.tif" 

    print(f"output files: {output_file}\n\n")
    
    renames_file(output_file)
    renames_file(output_file_TIF)
    
    print(f"> Saving projections as:\n\t--> {output_file}\n\t--> {output_file_TIF} \n")
    np.save(output_file, im_2d)
    imsave(output_file_TIF, im_2d)


def projects_3D_volumes(im, z_min = 0, z_max = np.inf):

    if z_max == np.inf:
        z_max = im.shape[0]
        
    print(f"\n> projecting image from {z_min} to {z_max} using max projection")
    im_2d = np.max(im[z_min:z_max,:,:], axis=0)
    
    return im_2d


def remove_small_masks(im0, num_pixels_min =500, z_min = 0, z_max = np.inf):

    if z_max == np.inf:
        z_max = im0.shape[0]

    im = im0[z_min:z_max,:,:]        
    regions = regionprops(im)
    
    n_masks = len(regions)
    
    print(f"$ Number of masks identified: {n_masks}")
    
    print("$ Iterating and removing masks with less than {} pixels...".format(num_pixels_min))
    
    im_clean = remove_small_objects(im0, min_size = num_pixels_min)
    
    regions_new = regionprops(im_clean[z_min:z_max,:,:])
    n_masks_new = len(regions_new)
    
    print(f"$ Number of masks left: {n_masks_new}")

    return im_clean

def process_images(p):

   files = p['files']

   if len(files) > 0 and files[0] is not None:

       print("\n{} files to process= <{}>".format(len(files), "\n".join(map(str, files))))

       # iterates over traces in folder
       for file in files:
  
            filename, file_ext = os.path.splitext(file)
            # = file.split('.')[1]
                        
            if file_ext == '.tif' or file_ext == '.tiff':
                im = imread(file)
            elif file_ext == '.npy':
                im = np.load(file)

            if p["convert"]:
                output_file = filename
                print(f"> Converting {filename} with extension: {file_ext}")
                if file_ext == '.tif' or file_ext == '.tiff':
                    np.save(output_file+'.npy', im)         
                    print(f"> Saved 3D image as \n\t--> {filename}.npy")
                elif file_ext == '.npy':
                    imsave(output_file+'.tif', im)
                    print(f"> Saved 3D image as \n\t--> {filename}.tif")

            print(f"> Analyzing image {file} with extension: {file_ext}")
            
            # removes small masks
            im_clean = remove_small_masks(im, num_pixels_min = p["num_pixels_min"], z_min = p['z_min'], z_max = p['z_max'])
            
            # projects labeled image
            im_2d = projects_3D_volumes(im_clean, z_min = p['z_min'], z_max = p['z_max'])
            
            # saves projections
            save_projections(file, im_2d)
            
            if p["save"]:
                output_file = filename + "_filtered" +  file_ext                                     
                if file_ext == '.tif' or file_ext == '.tiff':
                    imsave(output_file, im_clean)
                elif file_ext == '.npy':
                    np.save(output_file, im_clean)         
                print(f"> Saved 3D image as \n\t--> {output_file}")
   else:
       print("! Error: did not find any file to analyze. Please provide one using --input or --pipe.")


def main():

    # [parsing arguments]
    p = parse_arguments()

    # [loops over lists of datafolders]
    process_images(p)
        
    print("Finished execution")
    

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    main()
