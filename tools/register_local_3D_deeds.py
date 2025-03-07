#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script processes 3D registration of files using the register_3D_deeds_blocks.py script.
It allows processing all files or specific files based on user input.

Usage:
    Process all files:
        register_local_3D_deeds.py -a -r reference_file.tif

    Process specific files by barcodes:
        register_local_3D_deeds.py -b 16 18 20 -r reference_file.tif

Arguments:
    -a, --all                       Process all files matching the pattern "scan_001_RT*_???_ROI_converted_decon_ch00.tif".
    -b, --barcodes_to_register      List of barcodes to register (e.g., 16 18 20).
    -r, --reference_file            Reference file for the registration (required).
"""

import os, sys
import json
import re
import subprocess
import argparse
import concurrent.futures

def read_shifts(file_path):
    """Read the shift dictionary from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"! Shift file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"! Error decoding JSON file: {file_path}")
        return None

def check_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def process_files(files, reference_file, shifts_dict, args):
    for file in files:
        # Extract the base filename without the directory path
        base_filename = os.path.basename(file)
        
        # Extract ROI and cycle from the filename
        regex_pattern = r'scan_(?P<runNumber>[0-9]+)_(?P<cycle>[a-zA-Z0-9]+)_(?P<roi>[0-9]+)_ROI_converted_decon_(?P<channel>ch00)\.(tif|nii|nii.gz|h5|hdf5)'
        match = re.match(regex_pattern, base_filename)
        if not match:
            print(f"! Filename does not match pattern: {base_filename}")
            continue

        roi = f"ROI:{match.group('roi')}"
        cycle = match.group('cycle')

        # Get the shift values from the dictionary
        try:
            shifts = shifts_dict.get(roi, {}).get(cycle, None)
        except AttributeError:
            sys.exit(f'Error, could not find shift for {roi}:{cycle} ')
            
        # checks folders
        if args.folder == './':
            output_folder = 'register_local'    
        else:
            output_folder = args.folder.rstrip('/')+os.sep+'register_local'
            
        data_folder = output_folder+os.sep+'data'
        check_and_create_folder(output_folder)
        check_and_create_folder(data_folder)

        # Construct the displacement field filename
        displacement_field = data_folder+os.sep+f"{base_filename[:-4]}_DF.h5"
        
        # Construct the output filename
        output_file = data_folder+os.sep+f"{base_filename[:-4]}_aligned.tif"
        
        # Construct the log filename
        log_file = data_folder+os.sep+f"{base_filename[:-4]}_DF.log"
        
        # Run the register_3D_deeds_blocks.py command with the constructed arguments and log the output
        with open(log_file, "w") as log:
            cmd = [
                "register_3D_deeds_blocks.py",
                "--reference", reference_file,
                "--moving", file,
                "--displacement_field", displacement_field,
                "--png_folder", output_folder,
                "--output", output_file
            ]
            
            if shifts:
                cmd += ["--shifts", str(shifts[0]), str(shifts[1])," 0 "]
                print(f"$ Applying shifts for {base_filename}: {shifts}")
            else:
                print(f"! No shifts found for {base_filename}")

            print("="*80)
            print(f"$ will run: \n{' '.join(cmd)}")
            
            subprocess.run(cmd,
                stdout=log,
                stderr=subprocess.STDOUT
            )

def process_single_file(file, reference_file, shifts_dict, args):
    # Extract the base filename without the directory path
    base_filename = os.path.basename(file)
    
    # Extract ROI and cycle from the filename
    regex_pattern = r'scan_(?P<runNumber>[0-9]+)_(?P<cycle>[a-zA-Z0-9]+)_(?P<roi>[0-9]+)_ROI_converted_decon_(?P<channel>ch00)\.(tif|nii|nii.gz|h5|hdf5)'
    match = re.match(regex_pattern, base_filename)
    if not match:
        print(f"! Filename does not match pattern: {base_filename}")
        return

    roi = f"ROI:{match.group('roi')}"
    cycle = match.group('cycle')

    # Get the shift values from the dictionary
    shifts = shifts_dict.get(roi, {}).get(cycle, None)
    
    # checks folders
    if args.folder == './':
        output_folder = 'register_local'    
    else:
        output_folder = args.folder.rstrip('/')+os.sep+'register_local'
        
    data_folder = output_folder+os.sep+'data'
    check_and_create_folder(output_folder)
    check_and_create_folder(data_folder)

    # Construct the displacement field filename
    displacement_field = data_folder+os.sep+f"{base_filename[:-4]}_DF.h5"
    
    # Construct the output filename
    output_file = data_folder+os.sep+f"{base_filename[:-4]}_aligned.tif"
    
    # Construct the log filename
    log_file = data_folder+os.sep+f"{base_filename[:-4]}_DF.log"
    
    # Run the register_3D_deeds_blocks.py command with the constructed arguments and log the output
    with open(log_file, "w") as log:
        cmd = [
            "register_3D_deeds_blocks.py",
            "--reference", reference_file,
            "--moving", file,
            "--displacement_field", displacement_field,
            "--png_folder", output_folder,
            "--output", output_file
        ]
        
        if shifts:
            cmd += ["--shifts", str(shifts[0]), str(shifts[1]), "0"]
            print(f"$ Applying shifts for {base_filename}: {shifts}")
        else:
            print(f"! No shifts found for {base_filename}")

        print("="*80)
        print(f"$ will run: \n{' '.join(cmd)}")
        
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

def process_files_parallel(files, reference_file, shifts_dict, args):
    # Use ThreadPoolExecutor to run multiple processes in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks to the executor for each file
        futures = [executor.submit(process_single_file, file, reference_file, shifts_dict, args) for file in files]
        
        # Wait for all futures to complete and handle exceptions if needed
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Get the result to check for any exceptions
            except Exception as e:
                print(f"Error processing file: {e}")


def find_files(pattern, folder):
    files = [f for f in os.listdir(folder) if re.match(pattern, f)]
    
    if len(files) == 0:
        print(f"! Could not find files to process in the folder: {folder}")
    else: 
        print(f"$ Found {len(files)} files to process")
        
    return files

def main():
    parser = argparse.ArgumentParser(description="Process 3D registration of files.")
    parser.add_argument('--folder', required=True, help='Path to the folder containing files.')
    parser.add_argument('-a', '--all', action='store_true', help='Process all files matching the pattern "_RT*_".')
    parser.add_argument('-b', '--barcodes_to_register', nargs='+', help='List of barcodes to register (e.g., 16 18 20).')
    parser.add_argument('--reference_file', required=True, help='Reference file for the registration.')
    parser.add_argument('--channel', default='ch00', help='Channel to process. Default: ch00')
    parser.add_argument('--shifts_file', default='register_global/data/shifts.json', help='Path to the JSON file containing pre-computed shifts.')
    parser.add_argument("--parallel", help="Runs processes in parallel", action='store_true')

    args = parser.parse_args()

    reference_file = args.reference_file

    regex_pattern_RT = r'scan_(?P<runNumber>[0-9]+)_(?P<cycle>RT[0-9]+)_(?P<roi>[0-9]+)_ROI_converted_decon_(?P<channel>ch00)\.(tif|nii|nii.gz|h5|hdf5)'
    regex_pattern_other = r'scan_(?P<runNumber>[0-9]+)_(?P<cycle>[a-zA-Z0-9]+)_(?P<roi>[0-9]+)_ROI_converted_decon_(?P<channel>ch00)\.(tif|nii|nii.gz|h5|hdf5)'

    shifts_dict = read_shifts(args.shifts_file)
    
    if args.all:
        # Find all files matching the RT* pattern with any number between 000 and 999 before 'ROI'
        files_RT = find_files(regex_pattern_RT, args.folder)
        files_other = find_files(regex_pattern_other, args.folder)
        files = set(files_RT + files_other)
        print(f"$ will process {len(files)} unique files")
        
    elif args.barcodes_to_register:
        # Initialize an empty list of files
        files = []
        
        # Iterate through each barcode provided by the user
        for number in args.barcodes_to_register:
            # Construct the file pattern using the provided number
            file_pattern = f"scan_001_RT{number}_???_ROI_converted_decon_ch00.tif"
            
            # Find files matching the pattern and add to the list of files
            files.extend(find_files(file_pattern, args.folder))
    else:
        parser.error('No action requested, add -a or -b')

    files_to_process = []
    for file in files:
        if file != reference_file:
            match_RT = re.match(regex_pattern_RT, file)
            match_other = re.match(regex_pattern_other, file)
            
            if match_RT:
                match = match_RT
            elif match_other:   
                match = match_other

            cycle = match.group('cycle')
            channel = match.group('channel')

            if channel == args.channel:
                files_to_process.append(file)  
                print(f"$ Decoded filename: channel {channel}: cycle:{cycle}")

    # Process the found files
    if args.parallel:
        process_files_parallel(files_to_process, reference_file, shifts_dict, args)
    else:
        process_files(files_to_process, reference_file, shifts_dict, args)
    
if __name__ == "__main__":
    main()
