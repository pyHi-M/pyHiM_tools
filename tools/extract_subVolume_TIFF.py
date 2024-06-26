#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 10:42:15 2022

makes a subvolume extration from a tiff file


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
    parser.add_argument("-Z", "--zoom", help="provide zoom factor")
    parser.add_argument("-A", "--half", help="half the image is blanked out", action='store_true')
    parser.add_argument("-Sx", "--shift_x", help="Shifts image in x direction, provide number of pixels")
    parser.add_argument("-Sy", "--shift_y", help="Shifts image in y direction, provide number of pixels")
    
    args = parser.parse_args()

    p = {}
    p['needs_shift']=False
    
    if args.input:
        p["file"]= args.input
    else:
        print("Need to provide a filename!")
        sys.exit(-1)

    if args.zoom:
        p["zoom"]= int(args.zoom)
    else:
        p["zoom"]=10

    if args.shift_x:
        p["shift_x"]= float(args.shift_x)
    else:
        p["shift_x"]= 0.0

    if args.shift_y:
        p["shift_y"]= float(args.shift_y)
        p['needs_shift']=True
    else:
        p["shift_y"]= 0.0
        
    if args.half:
        p["half"]= args.half
        p['needs_shift']=True
    else:
        p["half"]= False
        
    print("$ zoom used: {}".format(p["zoom"]))

    return p

if __name__ == "__main__":

    # parameters
    p = parseArguments()

    zoom_factor = p["zoom"]
    file = p["file"]
    
    # loads TIFF files
    print(f"$ loading file: {file}")
    imageRaw = io.imread(file).squeeze()
    
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
    print(f"$ output image size: {im_size_out}")
    
    # blanks out half image
    if p['half']:
        template = np.zeros((im_size_out[1],im_size_out[2]))
        for i in range(im_size_out[1]):
            for j in range(int(im_size_out[2]/2),im_size_out[2]):
                template[i,j]=1.0 

        for z in range(im_size_out[0]):
            im[z,:,:] = im[z,:,:]*template 
    
    # shifts image
    if p['needs_shift']:
        shift = np.zeros((3))
        shift[0], shift[1], shift[2] = 0, p["shift_x"], p["shift_y"]
        im = shift_image(im, shift)
    
    # saves output data
    output_TIFF = file.split('.tif')[0] + '_zoom:'+str(p["zoom"])+'.tif'
    
    print(f"$ output file: {output_TIFF}")
    
    imsave(output_TIFF,im)
    
    plt.imshow(np.sum(100*im,axis=0),cmap='Reds')
    

