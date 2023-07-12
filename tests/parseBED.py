#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:35:56 2023

@author: chatGPT + marcnol

"""

def parse_bed_file(file_path):
    bed_dict = {}

    with open(file_path, 'r') as bed_file:
        for line in bed_file:
            if line.startswith('#'):
                continue  # Skip comment lines, if any

            fields = line.strip().split('\t')
        
            print(fields)
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

# Example usage
bed_file_path = "/home/marcnol/Dropbox/projects/methodological/oligopaint_design/projects/DM/Design_Julian_dec18/" + "3R_All_barcodes.bed"

parsed_bed = parse_bed_file(bed_file_path)

# Accessing the parsed BED data
for feature_name, features in parsed_bed.items():
    print(f'feature_name: {feature_name}')
    for feature in features:
        print(f"Start: {feature['start']}, End: {feature['end']}, chromosome: {feature['chromosome']}")


