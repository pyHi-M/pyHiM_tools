#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Organize files into ROI folders based on filename pattern and create symbolic links.

This script is used as a pre-processing step before launching pyHiM in separate folders.

Explanation:

    Default input_dir:
        Modified the argparse.ArgumentParser setup to make input_dir an optional positional argument (nargs='?') with a default value of '.' (current directory).
        If no directory is provided as an argument when running the script, it defaults to the current directory ('.').

    Handling Default Directory:
        os.path.abspath(args.input_dir) ensures that the provided or default directory path is converted to an absolute path for consistency.

    Output and Error Handling:
        The script still checks for the existence of parameters.json and reports errors if it is not found in the specified or default directory.

Usage:

    Save the updated script to a file (e.g., organize_files.py).
    Run the script from the command line without specifying a directory to use the current directory by default:
"""
    
import os
import re
import shutil
import json
import argparse

# Function to extract ROI from file name
def extract_roi(file_name):
    pattern = r"scan_(?P<runNumber>[0-9]+)_(?P<cycle>[\w|-]+)_(?P<roi>[0-9]+)_ROI_converted_decon_(?P<channel>[\w|-]+).tif"
    match = re.match(pattern, file_name)
    if match:
        return match.group('roi')
    return None

# Main function to sort files and organize them into ROI folders
def organize_files(input_dir,output_dir):
    # Validate input directory
    if not os.path.isdir(input_dir):
        raise ValueError(f"{input_dir} is not a valid directory.")
    if not os.path.isdir(output_dir):
        raise ValueError(f"{output_dir} is not a valid directory.")

    # Check if 'parameters.json' exists
    parameters_file = os.path.join(input_dir, 'parameters.json')
    if not os.path.exists(parameters_file):
        print(f"Error: {parameters_file} not found. Cannot proceed.")
        return

    # Create a set to store unique ROI values
    roi_set = set()

    # Iterate over files in input_dir
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.tif'):
            roi = extract_roi(file_name)
            if roi:
                roi_set.add(roi)

    # Create folders for each ROI and create symbolic links
    print(f"$ will sort {len(os.listdir(input_dir))} files in {len(roi_set)} folders")
    for roi in sorted(roi_set):
        roi_folder = os.path.join(output_dir, roi)
        os.makedirs(roi_folder, exist_ok=True)
        
        # Create symbolic links for each file with corresponding ROI
        for file_name in os.listdir(input_dir):
            if file_name.endswith('.tif'):
                if extract_roi(file_name) == roi:
                    source_file = os.path.join(input_dir, file_name)
                    link_name = os.path.join(roi_folder, file_name)
                    os.symlink(source_file, link_name, target_is_directory=False)

        # Create symbolic link for 'parameters.json' in each ROI folder
        parameters_link = os.path.join(roi_folder, 'parameters.json')
        os.symlink(parameters_file, parameters_link)

    print("$ Files sorted and symbolic links created successfully.")

# Entry point when script is run directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize files into ROI folders based on filename pattern and create symbolic links.")
    parser.add_argument("--input", nargs='?', default='.', help="Path to the input directory containing files to organize. Default is the current directory.")
    parser.add_argument("--output", nargs='?', default='.', help="Path to the output directory. Default is the current directory.")

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    organize_files(input_dir,output_dir)
