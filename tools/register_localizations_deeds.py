#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply deformation corrections to localizations based on barcode identity and deformation fields.

Usage:
    register_localizations_deeds.py --localizations localizations_3D_barcode_0.dat --deformation_folder /path/to/deformation_folder --output localizations_3D_barcode.dat

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
from tqdm import tqdm

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

def apply_deformation(localizations, deformation_field, barcode_id, z_binning=2, toleranceDrift_XY=1.0, toleranceDrift_Z = 3.0):
    """Apply deformation corrections to localizations."""
    
    print(f"$ applying registrations to {len(localizations)} localizations")

    counter=0
    for row in tqdm(localizations, desc="Processing localizations"):

        barcode_number=int(row["Barcode #"])
        if barcode_id == barcode_number:
            #print(f"$ Registering barcode: {barcode_number}")            
                
            x, y, z = int(row['xcentroid']), int(row['ycentroid']), int(z_binning*row['zcentroid'])
            
            # checks that the pixel fits in the volume of the DF vector field
            if 0 <= x < deformation_field.shape[2] and 0 <= y < deformation_field.shape[1] and 0 <= z < deformation_field.shape[0]:
                dz, dy, dx = deformation_field[z, y, x]
                
                # checks that the DF is not trying to correct more than the tolerance allows
                if np.abs(dx)<toleranceDrift_XY and np.abs(dy)<toleranceDrift_XY and np.abs(dz)<toleranceDrift_Z:
                    
                    row['xcentroid'] += -dx
                    row['ycentroid'] += -dy
                    row['zcentroid'] += -dz/z_binning
                    #print(f"$ Correcting {row['Buid']} barcode: {row['Barcode #']}| rxyz-final = ({row['xcentroid']},{row['ycentroid']},{row['zcentroid']}, dxyz={dx},{dy},{dz})")
                    counter+=1
                
    print(f"$ Corrected {counter} localizations.")
    
    return localizations


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Apply deformation corrections to localizations based on barcode identity and deformation fields.")
    parser.add_argument('--localizations', required=True, help='Path to the localizations file (Astropy format).')
    parser.add_argument('--deformation_folder', required=True, help='Path to the folder containing deformation fields.')
    parser.add_argument('--output', required=True, help='Path to the output file for corrected localizations.')
    parser.add_argument('--channel', default = 'ch00', help='Channel of the fiducial barcodes. Default = ch00')
    parser.add_argument('--zBinning', type=float, default=2.0, help='zBinning used in pyHiM. default=2.')
    parser.add_argument('--toleranceDrift_XY', default = 1.0 , type= float, help='Deformation field XY tolerance correction. Default = 1px')
    parser.add_argument('--toleranceDrift_Z', default = 3.0 , type= float, help='Deformation field Z tolerance correction. Default = 3px')    


    args = parser.parse_args()

    # Read the localizations
    localizations = read_localizations(args.localizations)

    # Regular expression to match the deformation files
    #regex_pattern = r'scan_(?P<runNumber>[0-9]+)_(?P<cycle>RT[0-9]+)_(?P<roi>[0-9]+)_ROI_converted_decon_(?P<channel>ch00)_DF\.(tif|nii|nii.gz|h5|hdf5)'
    regex_pattern = fr'scan_(?P<runNumber>[0-9]+)_(?P<cycle>RT[0-9]+)_(?P<roi>[0-9]+)_ROI_converted_decon_(?P<channel>{args.channel})_DF\.(tif|nii|nii.gz|h5|hdf5)'

    deformation_files = [f for f in os.listdir(args.deformation_folder) if re.match(regex_pattern, f)]

    if len(deformation_files) == 0:
        print(f"! Could not find deformation files in: {args.deformation_folder}")
        return
    else: 
        print(f"$ Found these files to process: {deformation_files}")
        
    for deformation_file in deformation_files:
        print("-"*80)
        match = re.match(regex_pattern, deformation_file)
        if match:
            cycle = match.group('cycle')
            channel = match.group('channel')
            print(f"$ Analyzing file: {deformation_file}")
            print("$ Decoded barcode: {} channel of DF: {}".format(int(cycle.split('RT')[1]), channel))
            
            if channel == args.channel and 'RT' in cycle:
                barcode_id = int(cycle.split('RT')[1])
                deformation_file_path = os.path.join(args.deformation_folder, deformation_file)
                
                if os.path.exists(deformation_file_path):
                    print(f"Applying deformation field: {deformation_file_path} for barcode ID: {barcode_id}")
                    deformation_field = read_deformation_field(deformation_file_path)
                    localizations = apply_deformation(localizations,\
                        deformation_field,\
                        barcode_id,\
                        z_binning=args.zBinning,\
                        toleranceDrift_XY=args.toleranceDrift_XY,\
                        toleranceDrift_Z = args.toleranceDrift_Z)
                    
                else:
                    print(f"Deformation field not found: {deformation_file_path}")

    # Save the corrected localizations
    write_localizations(localizations, args.output)
    print(f"Corrected localizations saved to {args.output}")

if __name__ == "__main__":
    main()
