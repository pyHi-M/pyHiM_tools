#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 18:05:08 2023

@author: marcnol

Usage:
    
    $ ls scan_*ROI.tif | deinterleave_channels.py --pipe --N_channels 2

or just for a single file

    deinterleave_channels.py --input scan_001_DAPI_001_ROI.tif --N_channels 2

"""

from tifffile import imread, imsave
import numpy as np
import select
import argparse
import sys

# =============================================================================
# FUNCTIONS
# =============================================================================q


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Name of input trace file.")
    parser.add_argument("--N_channels", help="Number of channels in image.")
    parser.add_argument("--pipe", help="inputs Trace file list from stdin (pipe)", action="store_true")

    p = {}

    args = parser.parse_args()


    if args.input:
        p["input"] = args.input
    else:
        p["input"] = None


    if args.N_channels:
        p["N_channels"] = int(args.N_channels)
    else:
        p["N_channels"] = 2
        print('Assuming {} channels as default'.format(p["N_channels"]))

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

def load_stack(im, total_number_channels, selected_channel):
    """ Load images corresponding to a single channel from a multi channel stack.

    @param path: path to the selected stack
    @param total_number_channels: (int) total number of channels composing the stack
    @param selected_channel: (int) indicate the channel to extract
    @return: a stack of images corresponding to the selected channel
    """
    im = im[np.arange(start=selected_channel, stop=im.shape[0], step=total_number_channels, dtype=int), :, :]
    return im
        
        
def process_images(files=list(),N_channels=2):
   """
   Processes list of trace files and sends each to get analyzed individually

   Parameters
   ----------
   folder : TYPE
       DESCRIPTION.
   trace_files : TYPE, optional
       DESCRIPTION. The default is list().

   Returns
   -------
   None.

   """

   if len(files) > 0 and files[0] is not None:

       print("\n{} files to process= <{}>".format(len(files), "\n".join(map(str, files))))

       # iterates over traces in folder
       for file in files:

           print(f"> Analyzing image {file}")

           im = imread(file)

           for channel in range(N_channels):
               print(f"> Extracting channel: {channel}")

               im_single_channel = load_stack(im, N_channels, channel)           
           
               output = file.split('.')[0] + '_ch0' + str(channel) + '.tif'
               imsave(
                    output,
                    im_single_channel,
                    imagej=True,
                )
               
               print(f">>> Image saved: {output}")               

   else:
       print("! Error: did not find any file to analyze. Please provide one using --input or --pipe.")
       
# =============================================================================
# MAIN
# =============================================================================


def main():

    # [parsing arguments]
    p = parseArguments()

    # [loops over lists of datafolders]
    process_images(files=p["files"], N_channels=p["N_channels"])

    print("Finished execution")


if __name__ == "__main__":
    main()       
        
        