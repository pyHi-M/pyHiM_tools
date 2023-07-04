#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 13:26:12 2023

@author: marcnol
"""
from tifffile import imread, imsave
from matplotlib.pylab import plt
import matplotlib
import numpy as np
from skimage.measure import regionprops

font = {"weight": "normal", "size": 22}

matplotlib.rc("font", **font)

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
        
def analyze_masks_z(im,output = 'tmp.png'):

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
    min_area,max_area,mean_equivalent_diameter_area = list(), list(), list()
    for plane in range(N_planes):
        region = regionprops(im[plane,:,:])
        areas, equivalent_diameter_areas =[], []
        for idx in range(len(region)):
            areas.append(region[idx].area)
            equivalent_diameter_areas.append(region[idx].equivalent_diameter_area)
        min_area.append(np.min(areas))
        max_area.append(np.max(equivalent_diameter_areas))
        mean_equivalent_diameter_area.append(np.mean(equivalent_diameter_areas))
        
    datasets.append(mean_equivalent_diameter_area)
    xlabels.append('z, planes')
    ylabels.append('mean diameter, px')
    titles.append('mean diam distribution')

    datasets.append(min_area)
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

        
def process_images(files=list()):


   if len(files) > 0 and files[0] is not None:

       print("\n{} files to process= <{}>".format(len(files), "\n".join(map(str, files))))

       # iterates over traces in folder
       for file in files:

            print(f"> Analyzing image {file}")
            
            im = imread(file)
            
            output = file.split('.')[0] + '_z_mask_distribution' + '.png'

            analyze_masks_z(im,output=output) 
              
            print(f"\n>>> Image saved: {output}")               

   else:
       print("! Error: did not find any file to analyze. Please provide one using --input or --pipe.")
       


file = '/mnt/grey/DATA/rawData_2023/Experiment_58_Marie_Christophe_Christel_fly_embryo_wt/DAPI_tiff/denoised/' + 'scan_001_DAPI_001_ROI_nuclei_denoised_cp_masks.tif'

files=[file]

process_images(files)