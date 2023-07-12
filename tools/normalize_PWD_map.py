#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:45:17 2023

@author: marcnol

This script loads
    -a list of PWD maps in NPY format
    -a genomic distance map based on a list of barcodes, generated using `get_barcode_normalisation_map`
    -a CSV file with uniquebarcodes
    
from this it:
    - calculates the proximity frequency map
    - normalized it by the number of times two barcodes are found in a trace
    - calculates the power law decay
    - constructs the expected proximity map
    - gets the observed/expected proximity map 
    

"""
import numpy as np
import matplotlib.pylab as plt
import argparse
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import sys

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genomic_distance_map", help="Name of genomic distance matrix, in NPY format.")
    parser.add_argument("--input", help="Name of input PWD maps, in NPY format.")
    parser.add_argument("--file_output", help="Name of output files.")
    parser.add_argument("--proximity_threshold", help="proximity threshold in um")
    parser.add_argument(
        "-U", "--uniqueBarcodes", help="csv file with list of unique barcodes"
    )
    
    p = {}

    args = parser.parse_args()

    if args.genomic_distance_map:
        p["genomic_distance_map"] = args.genomic_distance_map
    else:
        p["genomic_distance_map"] = '/mnt/grey/DATA/rawData_2023/Experiment_72_Marie_Christophe_Christel_Droso_Late_embryos_confocal/Analysis/combined_3D_analysis/traces/' +"genomic_distance_map.npy"

    if args.uniqueBarcodes:
        p["uniqueBarcodes"] = args.uniqueBarcodes
    else:
        print(">> ERROR: you must provide a CSV file with the unique barcodes used")
        sys.exit(-1)
        
    if args.input:
        p["input"] = args.input
    else:
        p["input"] = '/mnt/grey/DATA/rawData_2023/Experiment_72_Marie_Christophe_Christel_Droso_Late_embryos_confocal/Analysis/combined_3D_analysis/traces/'"merged_traces_Matrix_PWDscMatrix.npy"

    if args.file_output:
        p["file_output"] = args.file_output
    else:
        p["file_output"] = p["input"].split('.')[0]

    if args.proximity_threshold:
        p["proximity_threshold"] = float(args.proximity_threshold)
    else:
        p["proximity_threshold"] = 0.3

    return p
    

def parse_bed_file(file_path):
    bed_dict = {}

    with open(file_path, 'r') as bed_file:
        for line in bed_file:
            if line.startswith('#'):
                continue  # Skip comment lines, if any

            fields = line.strip().split('\t')
        
            chromosome = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            feature_name = fields[3]

            # Create a dictionary entry for the feature
            if feature_name not in bed_dict:
                bed_dict[feature_name] = []

            bed_dict[feature_name].append({
                'start': start,
                'end': end,
                'chromosome': chromosome
            })

    return bed_dict

# Define the power-law function
def power_law(x, a, b):
    return a * np.power(x, b)



def normalize_contact_map(contact_map):
    n = contact_map.shape[0]
    normalized_map = np.zeros_like(contact_map)

    for i in range(n):
        for j in range(i, n):
            distance = abs(i - j)
            normalized_value = contact_map[i, j] / distance
            normalized_map[i, j] = normalized_value
            normalized_map[j, i] = normalized_value

    return normalized_map

def run_process(file_name='',genomic_distance_map='', file_barcodes='',file_output='',distance_threshold=.3):
    
    # reads barcodes used
    unique_barcodes = list(np.loadtxt(file_barcodes, delimiter=" "))
    unique_barcodes = [int(x) for x in unique_barcodes]
    N_barcodes = len(unique_barcodes)
    
    # loads genomic_distance_map
    gen_dist_matrix = np.load(genomic_distance_map)

    # loads PWD map
    PWD_map = np.load(file_name)

    # calculates contact map and normalizes it
    contact_map = PWD_map<distance_threshold
    contact_map_flat  = np.nansum(contact_map,axis=2, dtype=float) # /contact_map.shape[2]
    for i, barcode1 in enumerate(unique_barcodes):
        for j, barcode2 in enumerate(unique_barcodes):
            if i != j:
                contact_map_flat[i,j] = contact_map_flat[i,j]/np.count_nonzero(~np.isnan(PWD_map[i,j,:]))
            else:
                contact_map_flat[i,j] = 0

    # plots contact map
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(contact_map_flat, cmap='coolwarm', vmin=0,vmax=.1)        
    plt.xticks(np.arange(gen_dist_matrix.shape[0]), unique_barcodes)
    plt.yticks(np.arange(gen_dist_matrix.shape[0]), unique_barcodes)
    plt.title("Normalized proximity frequency map")
    plt.savefig(file_output+'_normalized_contact_map'+'.png')
    np.save(file_output+'_normalized_contact_map'+'.npy',contact_map_flat)        

    # calculated proximity frequency versus genomic distance
    gen_distance, contacts=[],[]
    for i, barcode1 in enumerate(unique_barcodes):
        for j, barcode2 in enumerate(unique_barcodes):
            if i != j:
                gen_distance.append(gen_dist_matrix[i,j])
                contacts.append(contact_map_flat[i,j])
   
    # Fit the power-law model to the data
    params, _ = curve_fit(power_law, gen_distance,contacts)

    # Extract the fitted parameters
    a_fit = params[0]
    b_fit = params[1]

    # Generate the fitted curve using the fitted parameters
    x_fit = np.linspace(min(gen_distance), max(gen_distance), 100)
    y_fit = power_law(x_fit, a_fit, b_fit)
    print(f"$ Fitted parameters: a = {a_fit}, b = {b_fit}")
    
    # Plot the original data and the fitted curve
    fig = plt.figure(figsize=(15, 15))
    plt.scatter(gen_distance,contacts, label='Data')
    plt.plot(x_fit, y_fit, 'r-', label='Fitted Curve')
    plt.xlabel('genomic distance, bp')
    plt.ylabel('proximity frequency')
    plt.legend()
    plt.title("Fit to power law")
    plt.savefig(file_output+'_power_law_fit'+'.png')

    # constructs expected contact map
    expected_map = np.zeros((N_barcodes,N_barcodes))
    
    for i, barcode1 in enumerate(unique_barcodes):
        for j, barcode2 in enumerate(unique_barcodes):
            if i != j:
                expected_map[i,j] = power_law(gen_dist_matrix[i,j], a_fit, b_fit) 
            else:
                expected_map[i,j] = np.nan
                
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(expected_map,cmap='coolwarm')
    plt.xticks(np.arange(gen_dist_matrix.shape[0]), unique_barcodes)
    plt.yticks(np.arange(gen_dist_matrix.shape[0]), unique_barcodes)
    plt.title("Expected proximity map")    
    plt.savefig(file_output+'_expected_proximity_map'+'.png')
    np.save(file_output+'_expected_proximity_map'+'.npy',expected_map)        

    # normalizes experimental map by expected map
    contact_map_flat = contact_map_flat/expected_map
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(contact_map_flat, cmap='coolwarm', vmin=0)        
    plt.xticks(np.arange(gen_dist_matrix.shape[0]), unique_barcodes)
    plt.yticks(np.arange(gen_dist_matrix.shape[0]), unique_barcodes)
    plt.title("Observed/expected proximity map")
    plt.savefig(file_output+'_observed_expected_proximity_map'+'.png')
    np.save(file_output+'_observed_expected_proximity_map'+'.npy',contact_map_flat)        

def main():

    # [parsing arguments]
    p = parseArguments()

    genomic_distance_map = p["genomic_distance_map"]
    file_name = p["input"]
    file_output = p["file_output"]
    file_barcodes = p["uniqueBarcodes"]
    
    if os.path.exists(p["genomic_distance_map"]) and os.path.exists(p["input"]) and os.path.exists(p["uniqueBarcodes"]):
        run_process(file_name=file_name,
                    genomic_distance_map=genomic_distance_map,
                    file_barcodes=file_barcodes,
                    file_output=file_output,
                    distance_threshold=p["proximity_threshold"])
    else:
        print(f"! Error: input files do not exist !\n{genomic_distance_map}\n{file_name}\n{file_barcodes}")
        
    print("Finished execution")
    
if __name__ == "__main__":
    main()       


        
        
        
        
        
        
        
        
        
        