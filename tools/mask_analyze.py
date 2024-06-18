#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 13:26:12 2023

@author: marcnol

known problems with scikit-image<0.19:
    
    AttributeError: '<class 'skimage.measure._regionprops.RegionProperties'>' object has no attribute 'equivalent_diameter_area'

    solve by:
    
    $ conda install -c anaconda scikit-image
"""
from tifffile import imread
from matplotlib.pylab import plt
import matplotlib
import numpy as np
from skimage.measure import regionprops
import sys
import select
import argparse
import os

font = {"weight": "normal", "size": 22}

matplotlib.rc("font", **font)

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Name of input file.")
    parser.add_argument("--output", help="Name of output dataset file name.")
    parser.add_argument("--pipe", help="inputs file list from stdin (pipe)", action="store_true")

    args = parser.parse_args()

    p = {}

    if args.input:
        p["input"] = args.input
    else:
        p["input"] = None

    if args.output:
        p["output_dataset"] = args.output
    else:
        p["output_dataset"] = None

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

def plots(datasets,xlabels,ylabels,titles):

    number_plots = len(datasets)
    fig = plt.figure(constrained_layout=True)
    im_size = 10
    fig.set_size_inches((im_size * number_plots, im_size))
    gs = fig.add_gridspec(1, number_plots)
    axes = [fig.add_subplot(gs[0, i]) for i in range(number_plots)]
    
    for dataset, axis, xlabel, ylabel, title in zip(datasets, axes, xlabels, ylabels,titles):
        axis.plot(dataset,'o-')
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)   
        axis.set_title(title)
        
def analyze_masks_z(im,output = 'tmp.png', output_dataset = None):

    print('> Analyzing mask in z ... ')

    datasets, xlabels, ylabels, titles = list(), list(), list(), list() 
    N_planes, dim_y, dim_x = im.shape[0], im.shape[1], im.shape[2]

    # calculates distribution of masks in z
    N_nonzero_pixels = [np.count_nonzero(im[plane,:,:])/(dim_x*dim_y) for plane in range(N_planes) ]

    datasets.append(N_nonzero_pixels)
    xlabels.append('z, planes')
    ylabels.append('proportion of empty pixels')
    titles.append('z-axis masks distribution')

    # area equivalent_diameter_area
    min_area,max_area,mean_equivalent_diameter_area, all_areas = list(), list(), list(), list()
    for plane in range(N_planes):
        region = regionprops(im[plane,:,:])
        areas, equivalent_diameter_areas =[], []
        for idx in range(len(region)):
            areas.append(region[idx].area)
            equivalent_diameter_areas.append(region[idx].equivalent_diameter_area)
            all_areas.append(region[idx].equivalent_diameter_area)
        min_area.append(np.min(areas))
        max_area.append(np.max(equivalent_diameter_areas))
        mean_equivalent_diameter_area.append(np.mean(equivalent_diameter_areas))
        
    datasets.append(mean_equivalent_diameter_area)
    xlabels.append('z, planes')
    ylabels.append('mean diameter, px')
    titles.append('mean diam distribution')

    # datasets.append(min_area)
    xlabels.append('z, planes')
    ylabels.append('min area, px')
    titles.append('min area distribution')

    datasets.append(max_area)
    xlabels.append('z, planes')
    ylabels.append('max diameter, px')
    titles.append('max diameter distribution')
        
    # plots
    plots(datasets,xlabels,ylabels,titles)
    plt.savefig(output)

    # saves output datasets for further processing
    if output_dataset is not None:
        datasets.append(all_areas)
        output_data_filename = output+"_"+output_dataset+'.npz'
        np.savez(output_data_filename,datasets)
        print(f"$ output data saved to: {output_data_filename}")
        
def process_images(files=list(), output_dataset = None):

   if len(files) > 0 and files[0] is not None:

       print("\n{} files to process= <{}>".format(len(files), "\n".join(map(str, files))))

       # iterates over images in folder
       for file in files:

            print(f"> Analyzing image {file}")
            
            im = imread(file)
            
            output = file.split('.')[0] + '_mask_stats' + '.png'

            analyze_masks_z(im,output=output, output_dataset = output_dataset ) 
              
            print(f"\n>>> Analysis saved: {output}")               

   else:
       print("! Error: did not find any file to analyze. Please provide one using --input or --pipe.")
       
# =============================================================================
# MAIN
# =============================================================================


def main():

    # [parsing arguments]
    p = parseArguments()

    print("Remember to activate environment: conda activate aydin!\n")

    # [loops over lists of datafolders]
    process_images(files = p['files'],output_dataset =p['output_dataset'])
        
    print("Finished execution")

if __name__ == "__main__":
    main()       
    