#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply deformation corrections to localizations based on barcode identity and deformation fields.

Usage:
    python apply_deformation_to_localizations.py --localizations localizations.fits --deformation_folder /path/to/deformation_folder --output corrected_localizations.fits

Installation environment:

pip install astropy numpy simpleITK h5py

"""

import argparse
import os
import re
import numpy as np
import SimpleITK as sitk
import h5py
from astropy.table import Table

def read_localizations(file_path):
    """Read a localizations table from a file."""
    return Table.read(file_path, format='ascii.ecsv')

def write_localizations(table, file_path):
    """Write a localizations table to a file."""
    table.write(file_path, format='ascii.ecsv', overwrite=True)

def read_deformation_field(file_path):
    """Read a deformation field from a file."""
    if file_path.endswith('.tif') or file_path.endswith('.tiff'):
        displacement_field = sitk.ReadImage(file_path)
    elif file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        displacement_field = sitk.ReadImage(file_path)
    elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
        with h5py.File(file_path, 'r') as h5_file:
            displacement_array = h5_file['image'][:]
            spacing = tuple(h5_file.attrs['spacing'])
            origin = tuple(h5_file.attrs['origin'])
            direction = tuple(h5_file.attrs['direction'])
        displacement_field = sitk.GetImageFromArray(displacement_array, isVector=True)
        displacement_field.SetSpacing(spacing)
        displacement_field.SetOrigin(origin)
        displacement_field.SetDirection(direction)
    else:
        raise ValueError("Unsupported file format for deformation field.")
    
    displacement_array = sitk.GetArrayFromImage(displacement_field)
    return displacement_array

import tqdm

def apply_deformation(localizations, deformation_field, barcode_id, z_binning=2):
    """Apply deformation corrections to localizations."""
    
    #corrected_localizations = localizations.copy()
    
    print(f"$ applying registrations to {len(localizations)} localizations")

    for row in localizations:
        barcode_number=int(row["Barcode #"])
        if barcode_id == barcode_number:
            print(f"$ Registering barcode: {barcode_number}")            
                
            x, y, z = int(row['xcentroid']), int(row['ycentroid']), int(z_binning*row['zcentroid'])
            if 0 <= x < deformation_field.shape[2] and 0 <= y < deformation_field.shape[1] and 0 <= z < deformation_field.shape[0]:
                dz, dy, dx = deformation_field[z, y, x]
                row['xcentroid'] += -dx
                row['ycentroid'] += -dy
                row['zcentroid'] += -dz/z_binning
                print(f"$ Correcting {row['Buid']} barcode: {row['Barcode #']}| rxyz-final = ({row['xcentroid']},{row['ycentroid']},{row['zcentroid']}, dxyz={dx},{dy},{dz})")
                    
                 
    return localizations
'''
    for idx,localizations_barcode in enumerate(loc_table_indexed.groups):
        barcode_number=int(list(set(localizations_barcode["Barcode #"]))[0])
        if barcode_id == barcode_number:
            print(f"$ Registering barcode: {barcode_number}")            
                
            for i, loc in enumerate(localizations_barcode):
                x, y, z = int(loc['xcentroid']), int(loc['ycentroid']), int(z_binning*loc['zcentroid'])
                if 0 <= x < deformation_field.shape[2] and 0 <= y < deformation_field.shape[1] and 0 <= z < deformation_field.shape[0]:
                    dz, dy, dx = deformation_field[z, y, x]
                    localizations_barcode['xcentroid'][i] += -dx
                    localizations_barcode['ycentroid'][i] += -dy
                    localizations_barcode['zcentroid'][i] += -dz/z_binning
                    print(f"$ Correcting {loc['Buid']} barcode: {loc['Barcode #']}| rxyz-final = ({localizations_barcode['xcentroid'][i]},{localizations_barcode['ycentroid'][i]},{localizations_barcode['zcentroid'][i]}, dxyz={dx},{dy},{dz})")
'''

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Apply deformation corrections to localizations based on barcode identity and deformation fields.")
    parser.add_argument('--localizations', required=True, help='Path to the localizations file (Astropy format).')
    parser.add_argument('--deformation_folder', required=True, help='Path to the folder containing deformation fields.')
    parser.add_argument('--output', required=True, help='Path to the output file for corrected localizations.')
    parser.add_argument('--zBinning', type=float, default=2.0, help='zBinning == pixelsize_z/pixelsize_xy. default=2.')

    args = parser.parse_args()

    # Read the localizations
    localizations = read_localizations(args.localizations)

    # Regular expression to match the deformation files
    regex_pattern = r'scan_(?P<runNumber>[0-9]+)_(?P<cycle>RT[0-9]+)_(?P<roi>[0-9]+)_ROI_converted_decon_(?P<channel>ch00)_DF\.(tif|nii|nii.gz|h5|hdf5)'

    deformation_files = [f for f in os.listdir(args.deformation_folder) if re.match(regex_pattern, f)]

    if len(deformation_files) == 0:
        print(f"! Could not find deformation files in: {args.deformation_folder}")
    else: 
        print(f"$ Found these files to process: {deformation_files}")
        
    for deformation_file in deformation_files:
        match = re.match(regex_pattern, deformation_file)
        if match:
            cycle = match.group('cycle')
            channel = match.group('channel')
            print(f"$ Analyzing file: {deformation_file}")
            print("$ Decoded barcode: {} channel of DF: {}".format(int(cycle.split('RT')[1]), channel))
            
            if channel == 'ch00' and 'RT' in cycle:
                barcode_id = int(cycle.split('RT')[1])
                deformation_file_path = os.path.join(args.deformation_folder, deformation_file)
                
                if os.path.exists(deformation_file_path):
                    print(f"Applying deformation field: {deformation_file_path} for barcode ID: {barcode_id}")
                    #loc_table_indexed = localizations.group_by("Barcode #")
                    #barcode_localizations = localizations[localizations['Barcode #'] == barcode_id]
                    deformation_field = read_deformation_field(deformation_file_path)
                    localizations = apply_deformation(localizations, deformation_field, barcode_id, z_binning=args.zBinning)
                    
                    # Update the original table with corrected localizations
                    '''
                    for row in corrected_localizations:
                        localizations[(localizations['Barcode #'] == barcode_id) & 
                                        (localizations['xcentroid'] == row['xcentroid']) &
                                        (localizations['ycentroid'] == row['ycentroid']) &
                                        (localizations['zcentroid'] == row['zcentroid'])] = row
                    '''
                else:
                    print(f"Deformation field not found: {deformation_file_path}")

    # Save the corrected localizations
    write_localizations(localizations, args.output)
    print(f"Corrected localizations saved to {args.output}")

if __name__ == "__main__":
    main()
