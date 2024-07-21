#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 10:42:15 2022

Shifts image by a given amount in XYZ

@author: marcnol
"""

# imports
import os, argparse, sys, glob
from skimage import io
from tifffile import imsave
import matplotlib.pylab as plt
import numpy as np
from scipy.ndimage import shift as shift_image


def parseArguments():
    # [parsing arguments]
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--input", help="folder TIFF filename ")
    parser.add_argument("--shift_x", help="Shifts image in x direction, provide number of pixels")
    parser.add_argument("--shift_y", help="Shifts image in y direction, provide number of pixels")
    parser.add_argument("--shift_z", help="Shifts image in z direction, provide number of pixels")
    
    args = parser.parse_args()

    p = {}
    p['needs_shift']=False
    
    if args.input:
        p["file"]= args.input
    else:
        print("Need to provide a filename!")
        sys.exit(-1)

    if args.shift_x:
        p["shift_x"]= float(args.shift_x)
    else:
        p["shift_x"]= 0.0

    if args.shift_y:
        p["shift_y"]= float(args.shift_y)
    else:
        p["shift_y"]= 0.0

    if args.shift_z:
        p["shift_z"]= float(args.shift_z)
    else:
        p["shift_z"]= 0.0
        
    return p

if __name__ == "__main__":

    # parameters
    p = parseArguments()
    print("This algorithm shifts a 3D volume in TIF format by a user-provided amount.")
    print("Use the shift values as produced by pyHiM or our other registation methods.")

    file = p["file"]
    
    # loads TIFF files
    print(f"$ loading file: {file}")
    imageRaw = io.imread(file).squeeze()
    
    im_size_out = imageRaw.shape
    print(f"$ output image size: {im_size_out}")
    
    # shifts image
    shift = np.zeros((3))
    shift[0], shift[1], shift[2] = p["shift_z"], p["shift_x"], p["shift_y"]
    print(f"$ shifts that will be applied (ZXY): {shift}")
    im = shift_image(imageRaw, shift)
    
    # saves output data
    output_TIFF = file.split('.tif')[0] + '_shifted' + '.tif'
    
    print(f"$ output file: {output_TIFF}")
    
    imsave(output_TIFF,im)
    
    plt.imshow(np.sum(100*im,axis=0),cmap='Reds')
    

