#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 08:52:38 2023

@author: marcnol

this script will load and plot histograms of nuclei diameters

"""
import glob, os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {"weight": "normal", "size": 22}
matplotlib.rc("font", **font)

folder = '/home/marcnol/grey/ProcessedData_2023/Experiment_62_Marie_Christophe_Christel_fly_embryo_mutant/DAPI_tiff'
threshold = 38

folder = '/home/marcnol/grey/ProcessedData_2023/Experiment_55_Christophe_NC14/DAPI_tif'
threshold = 44

#%%# opens npz
files = glob.glob(folder+os.sep+"*data.npz")
file=files[0]
nuclear_sizes=3 # this data is normally in the third position of the list

datasets={}
for i,file in enumerate(files):
    content=np.load(file, allow_pickle=True)
    variable_name=content.files[0]
    data = content[variable_name]
    data_prefiltered = np.array(data[nuclear_sizes])
    data_postfiltered = data_prefiltered[data_prefiltered>threshold]
    datasets[str(i)] = data_postfiltered

#%%
# plots histograms
fig = plt.figure(figsize=(12, 10))

for key in datasets.keys():
    if "1" not in key and "7" not in key and "4" not in key:
        ax=sns.kdeplot(datasets[key], shade=True, alpha=.2, label = key,   cumulative=True, common_grid=True)
        ax.set_xlim(30, 120)
plt.xlabel('diameter,px')
# ax.legend
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
  
#%%
# saves output figure

fig.savefig(folder+os.sep+'nuclear_diameters.png')
fig.savefig(folder+os.sep+'nuclear_diameters.svg')



