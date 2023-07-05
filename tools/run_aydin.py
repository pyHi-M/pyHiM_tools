#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 12:41:01 2023

@author: marcnol

Usage:
    
    $ ls scan_*ROI.tif | run_aydin.py --pipe 
    
or just for a single file

    run_aydin.py --input scan_001_DAPI_001_ROI.tif 

"""

import sys
import subprocess
import select
import argparse
import os

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Name of input trace file.")
    parser.add_argument("--dict_path", help="Path to the dictionary of aydin parameters generated using the GUI")
    parser.add_argument("--pipe", help="inputs Trace file list from stdin (pipe)", action="store_true")

    p = {}

    args = parser.parse_args()

    if args.input:
        p["input"] = args.input
    else:
        p["input"] = None

    if args.dict_path:
        p["dict_path"] = args.dict_path
    else:
        home_folder = os.environ.get('HOME')
        p["dict_path"] = home_folder + "/Repositories/pyHiM_tools/tools/aydin_dict.json"
        print("> Will use the dictionary of parameters from: {}".format(p["dict_path"]))

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
    
def run_aydin(image_path, dict_path):
    command = f"aydin denoise {image_path} --lower-level-args={dict_path}"
    subprocess.run(command, shell=True)

def process_images(files = list(), dict_path = '~/Repositories/pyHiM_tools/tools/aydin_dict.json'):
       
    print(f"Dict path: {dict_path}\n")
    
    if len(files) > 0:

        print("\n{} trace files to process= {}".format(len(files), "\n".join(map(str, files))))
           
        # iterates over traces in folder
        for file in files:
           
            print(f"> Denoising image {file}")
	
            run_aydin(file, dict_path)

# =============================================================================
# MAIN
# =============================================================================


def main():

    # [parsing arguments]
    p = parseArguments()

    print("Remember to activate environment: conda activate aydin!\n")

    if os.path.exists(p["dict_path"]):
    
        # [loops over lists of datafolders]
        process_images(files = p['files'],dict_path=p["dict_path"])
        
    else:
        print("! Sorry but the dictionary of parameters could not be found at {}".format(p["dict_path"]))
    
    print("Finished execution")

if __name__ == "__main__":
    main()       
        
 


