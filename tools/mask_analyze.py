#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script analyzes a mask file provided by the user and optionally the corresponding intensity image.
No changes are made to the mask file.

The script loads an input file with masks and plots:
- An image of the masks
- Equivalent diameter
- Mean diameter
- Mean area
- Maximum diameter
- Intensity histogram for masks (if intensity image is provided)

as a function of the z position.

Known problems with scikit-image < 0.19:
    AttributeError: '<class 'skimage.measure._regionprops.RegionProperties'>' object has no attribute 'equivalent_diameter_area'
    Solve by:
    $ conda install -c anaconda scikit-image

Usage:
    python analyze_mask.py --input input_file --output output_dataset [--intensity_image original_intensity_image]

Example:
    python analyze_mask.py --input mask.tif --output mask_analysis --intensity_image original.tif

Arguments:
    --input              Name of the input mask file (TIFF, NPY, or HDF5 format).
    --output             Name of the output dataset file name (optional).
    --intensity_image    Name of the original intensity image file (optional).
    --pipe               Inputs file list from stdin (pipe) (optional).

Installation:
    conda create -y -n mask_analysis python==3.11
    pip install numpy scipy h5py tifffile scikit-image matplotlib

Created on Tue Jul  4 13:26:12 2023
Author: marcnol
Updated by OpenAI's GPT-4

"""

from tifffile import imread
from matplotlib.pylab import plt
import matplotlib
import numpy as np
from skimage.measure import regionprops
from skimage import exposure
import sys
import select
import argparse
import h5py
from tqdm import trange

font = {"weight": "normal", "size": 22}
matplotlib.rc("font", **font)

def read_image(file_path):
    """Read an image from a file (TIFF, NPY, or HDF5)."""
    if file_path.endswith('.h5'):
        with h5py.File(file_path, 'r') as f:
            image = f['image'][:]
    elif file_path.endswith('.npy'):
        image = np.load(file_path)
    else:
        image = imread(file_path)
    return image

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Name of input file (TIFF, NPY, or HDF5 format).")
    parser.add_argument("--output", help="Name of output dataset file name.")
    parser.add_argument("--intensity_image", help="Name of the original intensity image file (optional).")
    parser.add_argument("--pipe", help="Inputs file list from stdin (pipe)", action="store_true")

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

    if args.intensity_image:
        p["intensity"] = args.intensity_image
    else:
        p["intensity"] = None

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

def random_label_cmap(N=255):
    """Create a random colormap for labels."""
    """from matplotlib.colors import ListedColormap"""

    from random import random
    from matplotlib.colors import ListedColormap
    
    colors = [(0, 0, 0)] + [(random(), random(), random()) for _ in range(N)]
    return ListedColormap(colors)

def plot_raw_images_and_labels(image, label, ax,percentile=99.9):
    """
    Parameters
    ----------
    image : numpy ndarray
        3D raw image.

    label : numpy ndarray
        3D labeled image.
    """
    from scipy import ndimage
    from scipy.ndimage import label as nd_label

    start_idx=0
    if image is not None:
        moy = np.mean(image, axis=0)
        high_val = np.percentile(moy, percentile)

        # Rescale intensity values to saturate high values
        moy = exposure.rescale_intensity(moy,  in_range=(0, high_val), out_range=(0, 1))
        ax[0].imshow(moy, cmap="jet", origin="lower")
        ax[0].set_title("Raw Image")
        start_idx=1
            
    #lbl_moy = label #np.max(label, axis=0)
    #lbl_moy, number_of_blobs = ndimage.label(label)
    #labeled_image, number_of_labels = nd_label(label)

    number_of_labels=np.max(label)
    cmap = random_label_cmap(N=number_of_labels + 1)

    ax[start_idx].imshow(label, cmap=cmap, origin="lower")
    ax[start_idx].set_title("Projected Labeled Image")

def plots(datasets, xlabels, ylabels, titles, mask_image=None, intensity_im=None):

    if mask_image is not None:
        if intensity_im is not None:        
            start_idx = 2
        else:
            start_idx = 1
    else:
        start_idx = 0

    number_plots = start_idx + 4 #len(datasets) + (1 if intensity_im is not None else 0)
    fig = plt.figure(constrained_layout=True)
    im_size = 10
    fig.set_size_inches((im_size * number_plots, im_size))
    gs = fig.add_gridspec(1, number_plots)
    axes = [fig.add_subplot(gs[0, i]) for i in range(number_plots)]

    if mask_image is not None:
        plot_raw_images_and_labels(intensity_im, mask_image, axes[:2])

    N_datasets_z = 2
    colors=['b','r','g','k']
    for idx, (dataset, xlabel, ylabel, title) in enumerate(zip(datasets[0:N_datasets_z], xlabels[0:N_datasets_z], ylabels[0:N_datasets_z], titles[0:N_datasets_z])):
        axis = axes[start_idx + idx]
        if isinstance(dataset, list) and all(isinstance(i, list) for i in dataset):            
            for i, d in enumerate(dataset):
                axis.plot(d, colors[i]+'o-')            
        else:
            axis.plot(dataset, 'o-')
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)

    # histogram of 3D pixels
    ax = axes[-2]
    ax.hist(datasets[-2], bins=50, log=True)
    ax.set_xlabel(xlabels[N_datasets_z])
    ax.set_ylabel(ylabels[N_datasets_z])
    ax.set_title(titles[N_datasets_z])

    if intensity_im is not None:
        ax = axes[-1]
        ax.hist(datasets[-1], bins=50, log=True)
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Frequency')
        ax.set_title('Intensity Histogram (log scale)')

def analyze_masks_z(im, intensity_im=None, output='tmp.png', output_dataset=None):
    print('> Analyzing mask in z ... ')

    datasets, xlabels, ylabels, titles = [], [], [], []
    N_planes, dim_y, dim_x = im.shape[0], im.shape[1], im.shape[2]

    # Calculate distribution of masks in z
    N_nonzero_pixels = [np.count_nonzero(im[plane, :, :]) / (dim_x * dim_y) for plane in range(N_planes)]

    datasets.append(N_nonzero_pixels)
    xlabels.append('z, planes')
    ylabels.append('proportion of empty pixels')
    titles.append('z-axis masks distribution')

    # Area equivalent_diameter_area
    min_area, max_area, mean_equivalent_diameter_area, all_areas = [], [], [], []
    for plane in trange(N_planes):
        region = regionprops(im[plane, :, :])
        areas, equivalent_diameter_areas = [], []
        for idx in range(len(region)):
            areas.append(region[idx].area)
            equivalent_diameter_areas.append(region[idx].equivalent_diameter_area)
            all_areas.append(region[idx].equivalent_diameter_area)
        min_area.append(np.min(areas) if areas else 0)
        max_area.append(np.max(equivalent_diameter_areas) if equivalent_diameter_areas else 0)
        mean_equivalent_diameter_area.append(np.mean(equivalent_diameter_areas) if equivalent_diameter_areas else 0)

    datasets.append([mean_equivalent_diameter_area,min_area,max_area])
    xlabels.append('z, planes')
    ylabels.append('max diameter, px')
    titles.append('mean/min/max diameters')

    # Calculate number of pixels histogram
    areas_3d = []
    region_3d = regionprops(im)
    for props in region_3d:
        areas_3d.append(props.area)
    datasets.append(areas_3d)
    xlabels.append('number of pixels')
    ylabels.append('Frequency')
    titles.append('Number of pixels')

    # Calculate intensity histogram
    mask_image = np.max(im, axis=0)
    if intensity_im is not None:
        intensities = []
        region_3d = regionprops(im,intensity_image=intensity_im)
        for props in region_3d:
            intensities.append(props.max_intensity)
        datasets.append(intensities)

    # Plot all datasets including mask image
    plots(datasets, xlabels, ylabels, titles, mask_image, intensity_im)
    plt.savefig(output)

    # Save output datasets for further processing
    if output_dataset is not None:
        datasets.append(all_areas)
        output_data_filename = output + "_" + output_dataset + '.npz'
        np.savez(output_data_filename, datasets)
        print(f"$ Output data saved to: {output_data_filename}")

def process_images(files=list(), intensity_file=None, output_dataset=None):
    if len(files) > 0 and files[0] is not None:
        print("\n{} files to process= <{}>".format(len(files), "\n".join(map(str, files))))

        intensity_im = None
        if intensity_file:
            intensity_im = read_image(intensity_file)

        # Iterates over images in folder
        for file in files:
            print(f"> Analyzing image {file}")
            im = read_image(file)
            output = file.split('.')[0] + '_mask_stats.png'
            analyze_masks_z(im, intensity_im=intensity_im, output=output, output_dataset=output_dataset) 
            print(f"\n>>> Analysis saved: {output}")
    else:
        print("! Error: did not find any file to analyze. Please provide one using --input or --pipe.")

def main():
    # [parsing arguments]
    p = parseArguments()

    # [loops over lists of data folders]
    process_images(files=p['files'], intensity_file=p['intensity'], output_dataset=p['output_dataset'])
        
    print("Finished execution")

if __name__ == "__main__":
    main()
