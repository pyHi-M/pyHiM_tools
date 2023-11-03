#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 09:19:08 2023

@author: marcnol
"""
from astropy.io import ascii
trace_file = "/mnt/grey/DATA/rawData_2022/Experiment_6_Olivier_Marie_seqRNA_HiM_Pancreas/Analysis_corrected_final/006_ROI/test/Trace_3D_barcode_mask:mask0_ROI:6_crap.ecsv"

trace_table = ascii.read(trace_file)  

label = 'crap'

#%% this keeps all spots that have the label

trace_table_new = trace_table.copy()

rows_to_remove = []

for idx, row in enumerate(trace_table_new):
    if label not in row['label']:
        print("$ tag: {}".format(row['label']))
        rows_to_remove.append(idx)

trace_table_new.remove_rows(rows_to_remove)

removed = len(trace_table)-len(trace_table_new)
print(f"$ Removed {removed} spots")

####

for row in trace_table_new:
    print("$ tag: {}".format(row['label']))
    
#%%
for row in trace_table:
    print("$ tag: {}".format(row['label']))

#%% this keeps all spots without the label

trace_table_new = trace_table.copy()

rows_to_remove = []

for idx, row in enumerate(trace_table_new):
    if label in row['label']:
        print("$ tag: {}".format(row['label']))
        rows_to_remove.append(idx)

trace_table_new.remove_rows(rows_to_remove)

removed = len(trace_table)-len(trace_table_new)
print(f"$ Removed {removed} spots")

for row in trace_table_new:
    print("$ tag: {}".format(row['label']))