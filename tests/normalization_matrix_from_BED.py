#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:45:17 2023

@author: marcnol

This script loads
    -a BED file with the coordinates of barcodes
    -a CSV file with unique barcodes 
    
from this it creates a matrix of genomic distances that are exported as PNG and as NPY

"""
import numpy as np
import matplotlib.pylab as plt
import argparse
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--barcodes_file_path", help="Name of input barcode list, in csv format.")
    parser.add_argument("--bed_file_path", help="Name of input barcode coordinates file, in bed format.")
    parser.add_argument("--file_output", help="Name of output files.")
    parser.add_argument("--file_PWD_map", help="Name of PWD maps, in NPY format.")

    p = {}

    args = parser.parse_args()

    if args.barcodes_file_path:
        p["barcodes_file_path"] = args.barcodes_file_path
    else:
        p["barcodes_file_path"] = '/mnt/grey/DATA/rawData_2023/Experiment_72_Marie_Christophe_Christel_Droso_Late_embryos_confocal/Analysis/combined_3D_analysis/traces/' +"merged_traces_Matrix_uniqueBarcodes.ecsv" 

    if args.bed_file_path:
        p["bed_file_path"] = args.bed_file_path
    else:
        p["bed_file_path"] = "/home/marcnol/Dropbox/projects/methodological/oligopaint_design/projects/DM/Design_Julian_dec18/" + "3R_All_barcodes.bed"
        
    if args.file_output:
        p["file_output"] = args.file_output
    else:
        p["file_output"] = '/mnt/grey/DATA/rawData_2023/Experiment_72_Marie_Christophe_Christel_Droso_Late_embryos_confocal/Analysis/combined_3D_analysis/traces/' +"genomic_distance_map" 

    if args.file_PWD_map:
        p["file_PWD_map"] = args.file_PWD_map
    else:
        p["file_PWD_map"] = '/mnt/grey/DATA/rawData_2023/Experiment_72_Marie_Christophe_Christel_Droso_Late_embryos_confocal/Analysis/combined_3D_analysis/traces/'"merged_traces_Matrix_PWDscMatrix.npy"



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

def run_process(barcodes_file_path,bed_file_path,file_output,file_contact_map,distance_threshold=.3):
    # reads BED and creates dict
    barcode_dict = parse_bed_file(bed_file_path)
    
    # reads barcodes used
    
    unique_barcodes = list(np.loadtxt(barcodes_file_path, delimiter=" "))
    unique_barcodes = [int(x) for x in unique_barcodes]
    
    # gets coordinates from each barcode
    barcode_unique_dict = {}
    for barcode in unique_barcodes:
        barcode_str='barcode_'+str(barcode)
        start_seq = int(barcode_dict[barcode_str][0]['start'])
        end_seq = int(barcode_dict[barcode_str][0]['end'])
        barcode_unique_dict[str(barcode)]=int((end_seq+start_seq)/2)
        
        
    # constructs genomic distance matrix
    N_barcodes = len(unique_barcodes)
    gen_dist_matrix = np.zeros((N_barcodes,N_barcodes))
    
    for i, barcode1 in enumerate(unique_barcodes):
        for j, barcode2 in enumerate(unique_barcodes):
            gen_dist_matrix[i,j] = np.abs( float( barcode_unique_dict[str(barcode1)] - barcode_unique_dict[str(barcode2)])) 
    
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(gen_dist_matrix, cmap='PiYG')        
    plt.xticks(np.arange(gen_dist_matrix.shape[0]), unique_barcodes)
    plt.yticks(np.arange(gen_dist_matrix.shape[0]), unique_barcodes)
    plt.title("Genomic distance map")
            
    plt.savefig(file_output+'_genomic_distance_map'+'.png')
    np.save(file_output+'_genomic_distance_map'+'.npy',gen_dist_matrix)        

    # loads PWD map
    PWD_map = np.load(file_contact_map)
    PWD_map_flat = np.nanmedian(PWD_map,axis=2)

    # calculates contact map and normalizes it
    contact_map = PWD_map<distance_threshold
    contact_map_flat  = np.nansum(contact_map,axis=2, dtype=np.float) # /contact_map.shape[2]
    for i, barcode1 in enumerate(unique_barcodes):
        for j, barcode2 in enumerate(unique_barcodes):
            if i != j:
                contact_map_flat[i,j] = contact_map_flat[i,j]/np.count_nonzero(~np.isnan(PWD_map[i,j,:]))
                # contact_map_flat[i,j] = contact_map_flat[i,j]
                # contact_map_flat[i,j] = np.count_nonzero(~np.isnan(PWD_map[i,j,:]))
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
    plt.title("Normalized proximity map")
    plt.savefig(file_output+'_normalized_proximity_map'+'.png')
    np.save(file_output+'_normalized_proximity_map'+'.npy',contact_map_flat)        

def main():
    

    # [parsing arguments]
    p = parseArguments()

    barcodes_file_path = p["barcodes_file_path"]
    bed_file_path = p["bed_file_path"]
    file_output = p["file_output"]
    file_contact_map = p["file_contact_map"]

    print("Remember to activate environment: conda activate aydin!\n")

    # if os.path.exists(p["dict_path"]):
    
        # [loops over lists of datafolders]
    run_process(barcodes_file_path,bed_file_path,file_output, file_contact_map)
        
    print("Finished execution")
    
if __name__ == "__main__":
    main()       


        
        
        
        
        
        
        
        
        
        