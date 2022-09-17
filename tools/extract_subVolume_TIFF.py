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

def parseArguments():
    # [parsing arguments]
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--file", help="folder TIFF filename ")
    parser.add_argument("-Z", "--zoom", help="provide zoom factor")

    args = parser.parse_args()

    p = {}

    if args.file:
        p["file"]= args.file
    else:
        print("Need to provide a filename!")
        sys.exit(-1)

    if args.zoom:
        p["zoom"]= int(args.zoom)
    else:
        p["zoom"]=10
        
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
    
    # makes subVolume extractions
    im_size = imageRaw.shape
    means = [int(x/2) for x in im_size]
    borders_left = [a-b//zoom_factor for a,b in zip(means,im_size)]
    borders_right = [a+b//zoom_factor for a,b in zip(means,im_size)]
    
    im = imageRaw[:,borders_left[1]:borders_right[1],borders_left[2]:borders_right[2]]
    
    # saves output data
    output_TIFF = file.split('.tif')[0] + '_zoom:'+str(p["zoom"])+'.tif'
    
    print(f"$ output file: {output_TIFF}")
    
    imsave(output_TIFF,im)
    
    plt.imshow(np.sum(100*im,axis=0),cmap='Reds')
    

