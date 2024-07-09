#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script processes 3D registration of TIFF files using the register_3D_deeds_blocks.py script.
It allows processing all files or specific files based on user input.

Usage:
    Process all files:
        python script_name.py -a -r reference_file.tif

    Process specific files by barcodes:
        python script_name.py -b 16 18 20 -r reference_file.tif

Arguments:
    -a, --all                       Process all files matching the pattern "scan_001_RT*_???_ROI_converted_decon_ch00.tif".
    -b, --barcodes_to_register      List of barcodes to register (e.g., 16 18 20).
    -r, --reference_file            Reference file for the registration (required).
"""

import os
import glob
import subprocess
import argparse

def find_files(pattern):
    # Use glob to find files matching the pattern
    return glob.glob(pattern)

def process_files(files, reference_file):
    for file in files:
        # Extract the base filename without the directory path
        base_filename = os.path.basename(file)
        
        # Construct the displacement field filename
        displacement_field = f"{base_filename[:-4]}_DF.h5"
        
        # Construct the output filename
        output_file = f"{base_filename[:-4]}_aligned.tif"
        
        # Construct the log filename
        log_file = f"{base_filename[:-4]}_DF.log"
        
        # Run the register_3D_deeds_blocks.py command with the constructed arguments and log the output
        with open(log_file, "w") as log:
            subprocess.run(
                [
                    "register_3D_deeds_blocks.py",
                    "--reference", reference_file,
                    "--moving", file,
                    "--displacement_field", displacement_field,
                    "--output", output_file
                ],
                stdout=log,
                stderr=subprocess.STDOUT
            )

def main():
    parser = argparse.ArgumentParser(description="Process 3D registration of files.")
    parser.add_argument('-a', '--all', action='store_true', help='Process all files matching the pattern "_RT*_".')
    parser.add_argument('-b', '--barcodes_to_register', nargs='+', help='List of barcodes to register (e.g., 16 18 20).')
    parser.add_argument('-r', '--reference_file', required=True, help='Reference file for the registration.')

    args = parser.parse_args()

    reference_file = args.reference_file

    if args.all:
        # Find all files matching the RT* pattern with any number between 000 and 999 before 'ROI'
        files = find_files("scan_001_RT*_???_ROI_converted_decon_ch00.tif")
    elif args.barcodes_to_register:
        # Initialize an empty list of files
        files = []
        
        # Iterate through each barcode provided by the user
        for number in args.barcodes_to_register:
            # Construct the file pattern using the provided number
            file_pattern = f"scan_001_RT{number}_???_ROI_converted_decon_ch00.tif"
            
            # Find files matching the pattern and add to the list of files
            files.extend(find_files(file_pattern))
    else:
        parser.error('No action requested, add -a or -b')
    
    # Process the found files
    process_files(files, reference_file)

if __name__ == "__main__":
    main()
