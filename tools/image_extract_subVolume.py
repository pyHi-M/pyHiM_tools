#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs subvolume extraction from a TIFF or HDF5 file,
with optional blanking out half of the image.

Usage:
    python image_extract_subVolume.py -F input_file -Z zoom_factor [-A] 

Example:
    python image_extract_subVolume.py -F input_image.tif -Z 10 -A 

Installation:
    conda create -y -n image_processing python==3.11
    pip install numpy scipy h5py tifffile scikit-image matplotlib

This script makes a subvolume extraction from a TIFF or HDF5 file with optional image manipulation.

Created on Sat Sep 10 10:42:15 2022
Updated by OpenAI's GPT-4

Author: marcnol
"""

# imports
import os, argparse, sys, glob
from skimage import io
from tifffile import imsave
import matplotlib.pylab as plt
import numpy as np
import h5py
import select

def check_file_existence(filename):
    """Check if the reference and moving image files exist."""
    if not os.path.exists(filename):
        print(f"Error: image file '{filename}' does not exist.")
        sys.exit(1)
        
def read_image(file_path):
    """Read an image from a file (TIFF or HDF5)."""
    if file_path.endswith('.h5'):
        with h5py.File(file_path, 'r') as f:
            image = f['image'][:]
    else:
        image = io.imread(file_path).squeeze()
    return image

def write_image(image, file_path):
    """Write an image to a file (TIFF or HDF5)."""
    if file_path.endswith('.h5'):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('image', data=image, compression='gzip')
    else:
        imsave(file_path, image)

def parseArguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--input", help="Input file (TIFF or HDF5 format)", required=True)
    parser.add_argument("-Z", "--zoom", help="Zoom factor", type=int, default=10)
    parser.add_argument("-A", "--half", help="Blank out half of the image", action='store_true')
    parser.add_argument("--pipe", help="Inputs file list from stdin (pipe)", action="store_true")

    args = parser.parse_args()

    p = {}
   
    p["file"] = args.input
    p["zoom"] = args.zoom
    p["half"] = args.half

    print(f"$ Zoom factor: {p['zoom']}")
    
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

def run(p,file):
    zoom_factor = p["zoom"]
    
    # loads image files
    check_file_existence(file)
    print(f"$ Loading file: {file}")
    imageRaw = read_image(file)
    
    if zoom_factor > 0:
        # makes subVolume extractions
        im_size = imageRaw.shape
        means = [int(x/2) for x in im_size]
        borders_left = [a-b//zoom_factor for a,b in zip(means,im_size)]
        borders_right = [a+b//zoom_factor for a,b in zip(means,im_size)]
        
        im = imageRaw[:,borders_left[1]:borders_right[1],borders_left[2]:borders_right[2]]
    else:
        im = imageRaw
        
    im_size_out = im.shape
    print(f"$ Output image size: {im_size_out}")
    
    # blanks out half image
    if p['half']:
        template = np.zeros((im_size_out[1],im_size_out[2]))
        for i in range(im_size_out[1]):
            for j in range(int(im_size_out[2]/2),im_size_out[2]):
                template[i,j] = 1.0 

        for z in range(im_size_out[0]):
            im[z,:,:] = im[z,:,:] * template 
    
    # saves output data
    output_file_ext = 'h5' if file.endswith('.h5') else 'tif'
    output_file = file.split('.')[0] + f'_zoom:{p["zoom"]}.{output_file_ext}'
    
    print(f"$ Output file: {output_file}")
    
    write_image(im, output_file)
    
    plt.imshow(np.sum(100*im, axis=0), cmap='Reds')
    plt.show()

if __name__ == "__main__":
    
    # parameters
    p = parseArguments()

    if len(files) > 0 and files[0] is not None:
        print("\n{} files to process= <{}>".format(len(files), "\n".join(map(str, files))))

        for file in files:
            run(p, file)
    
    else:
        print("! Error: did not find any file to analyze. Please provide one using --input or --pipe.")
