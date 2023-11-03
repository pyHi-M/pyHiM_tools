#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:09:51 2023

@author: marcnol
"""

folder= "/mnt/grey/DATA/rawData_2022/Experiment_6_Olivier_Marie_seqRNA_HiM_Pancreas/Analysis_corrected_final/006_ROI"
file=folder+'/scan_001_DAPI_006_ROI_converted_decon_ch00.tif'

import numpy as np
from cellpose import models
from cellpose.io import imread

def run_cellpose_api(image_path, diam, cellprob, flow, stitch,gpu = True,):
    
    # model_type='cyto' or 'nuclei' or 'cyto2'
    model = models.Cellpose(gpu = gpu, model_type='cyto')

    # list of files
    files = [image_path]

    imgs = [imread(f) for f in files]

    # define CHANNELS to run segementation on
    channels = [[0,0]]

    # runs model    
    masks, flows, styles, diams = model.eval(imgs,channels=channels,diameter=None,cellprob_threshold = cellprob,flow_threshold=flow,stitch_threshold= stitch,)

    return masks

masks=run_cellpose_api([file],50,-8,10,0,.1,gpu=True)

#%%