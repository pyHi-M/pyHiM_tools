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
import os, sys
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--barcodes_file_path", help="Name of input barcode list, in csv format.")
    parser.add_argument("--bed_file_path", help="Name of input barcode coordinates file, in bed format.")
    parser.add_argument("--file_output", help="Name of output files.")
    parser.add_argument("--shift_barcode_number", help="Shift between barcode names in BED file and barcode list.")

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
        p["file_output"] = "genomic_distance_map" 

    if args.shift_barcode_number:
        p["shift_barcode_number"] = int(args.shift_barcode_number)
    else:
        p["shift_barcode_number"] = 0
    return p
    

def parse_bed_file(file_path):
    bed_dict = {}
    
    with open(file_path, 'r') as bed_file:
        for line in bed_file:
            if line.startswith('#'):
                continue  # Skip comment lines, if any

            fields = line.strip().split('\t')
        
            chromosome = fields[0]
            keep_reading=True
            try:
                start = int(fields[1])
            except IndexError:
                keep_reading=False
            
            if keep_reading:
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
            else:
                print('>> Skipping header!')

    print(f"$ Assigned {len(bed_dict)} keys: {bed_dict.keys()}")
    return bed_dict


def run_process(barcodes_file_path='',bed_file_path='',file_output='',shift_barcode_number=0):
    # reads BED and creates dict
    barcode_dict = parse_bed_file(bed_file_path)
    
    # reads barcodes used
    unique_barcodes = list(np.loadtxt(barcodes_file_path, delimiter=" "))
    unique_barcodes = [int(x) for x in unique_barcodes]
    
    # gets coordinates from each barcode
    barcode_unique_dict = {}
    for barcode in unique_barcodes:
        barcode_str=str(barcode-shift_barcode_number)
        try:
            start_seq = int(barcode_dict[barcode_str][0]['start'])
        except KeyError:
            print(f'ERROR: Barcode <{barcode_str}> not found in BED file!')
            print(f'$ The barcode name in <{bed_file_path}> should be the same as in <{barcodes_file_path}> ;)')
            sys.exit(-1)
        end_seq = int(barcode_dict[barcode_str][0]['end'])
        barcode_unique_dict[str(barcode)]=int((end_seq+start_seq)/2)
        
    # constructs genomic distance matrix
    N_barcodes = len(unique_barcodes)
    gen_dist_matrix = np.zeros((N_barcodes,N_barcodes))
    
    for i, barcode1 in enumerate(unique_barcodes):
        for j, barcode2 in enumerate(unique_barcodes):
            gen_dist_matrix[i,j] = np.abs( float( barcode_unique_dict[str(barcode1)] - barcode_unique_dict[str(barcode2)])) 

    # makes figures    
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(gen_dist_matrix, cmap='PiYG')        
    plt.xticks(np.arange(gen_dist_matrix.shape[0]), unique_barcodes)
    plt.yticks(np.arange(gen_dist_matrix.shape[0]), unique_barcodes)

    # saves outputs
    plt.savefig(file_output+'_genomic_distances'+'.png')
    np.save(file_output+'_genomic_distances'+'.npy',gen_dist_matrix)        
    
def main():

    # [parsing arguments]
    p = parseArguments()

    barcodes_file_path = p["barcodes_file_path"]
    bed_file_path = p["bed_file_path"]
    file_output = p["file_output"]
    shift_barcode_number = p["shift_barcode_number"]
    
    print(f"$ input parameters: input \nbarcodes file: {barcodes_file_path}\nBED file: {bed_file_path}")

    if os.path.exists(p["barcodes_file_path"]) and os.path.exists(p["bed_file_path"]):
        run_process(barcodes_file_path=barcodes_file_path,
                    bed_file_path=bed_file_path,
                    file_output=file_output,
                    shift_barcode_number=shift_barcode_number)
    else:
        print(f"! Error: input files do not exist !\n{barcodes_file_path}\n{bed_file_path}")

    print("Finished execution")
    
if __name__ == "__main__":
    main()       


        
        
        
        
        
        
        
        
        
        