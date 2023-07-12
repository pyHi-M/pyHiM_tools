#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:45:32 2023

@author: marcnol
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pylab as plt

#%% original data - replicate 1
file_name = "/home/marcnol/Dropbox/Labbooks/Lab_book_marcnol/Projects/Pc_DM_late_embryo/data/Julian/pc_st15/all_embryos_v2_PAPER/buildsPWDmatrix/all_embryos_experiment_21_2021_SC_analysed_all.mat"

m = loadmat(file_name, mdict=None, appendmat=True)

#%%
barcodes = m['barcodes2plot']-1

sc_matrix_full=  m['SC_normalized_cumulative_matrix']
N_cells=sc_matrix_full.shape[2]
threshold=2

N_barcodes = barcodes.shape[1]
sc_matrix  = np.zeros((N_barcodes,N_barcodes,N_cells))
for i in range(N_barcodes):
    for j in range(N_barcodes):
        for k in range(N_cells):        
            index_i ,index_j = barcodes[0][i], barcodes[0][j]
            if sc_matrix_full[index_i ,index_j,k]< threshold:
                sc_matrix[i,j,k] = sc_matrix_full[index_i ,index_j,k]
            else:
                sc_matrix[i,j,k] = np.nan

sc_matrix_flat = np.nanmean(sc_matrix, axis=2)
plt.imshow(sc_matrix_flat,cmap='PiYG',vmin=0.5, vmax=1.5)

# plt.hist(sc_matrix_flat[18,:])

file_output = "/home/marcnol/Dropbox/Labbooks/Lab_book_marcnol/Projects/Pc_DM_late_embryo/data/Julian/pc_st15/all_embryos_v2_PAPER/buildsPWDmatrix/Pc_stg14_replicate1.npy"

np.save(file_output,sc_matrix)

#%% New data - replicate 2

file2 = '/mnt/grey/DATA/rawData_2023/Experiment_62_Marie_Christophe_Christel_fly_embryo_mutant/Analysis/combined_3D_analysis/traces/merged_traces_Matrix_PWDscMatrix.npy'

threshold=2

sc_matrix_full1= np.load(file2)
N_barcodes=sc_matrix_full1.shape[0]
N_cells=sc_matrix_full1.shape[2]
sc_matrix1  = np.zeros((N_barcodes,N_barcodes,N_cells))
for i in range(N_barcodes):
    for j in range(N_barcodes):
        for k in range(N_cells):        
            if sc_matrix_full1[i,j,k]< threshold:
                sc_matrix1[i,j,k] = sc_matrix_full1[i,j,k]
            else:
                sc_matrix1[i,j,k] = np.nan
        
sc_matrix_flat1 = np.nanmean(sc_matrix1,axis=2)
#plt.imshow(sc_matrix_flat1 ,cmap='PiYG',vmin=0, vmax=1.5)

dist = sc_matrix1[1,2,:]
# dist=dist[dist<2]
plt.hist(dist)

#%%

sc_matrix_flat1 = np.nanmean(sc_matrix1,axis=2)

# plt.imshow(sc_matrix_flat1)
plt.imshow(sc_matrix_flat1*1 ,cmap='PiYG', vmin=0.5,vmax=1.5)
# plt.imshow(sc_matrix_flat1 ,cmap='PiYG',vmin=0)

#%% data merge

sc_matrix_merge = np.concatenate((sc_matrix, sc_matrix1), axis=2)

sc_matrix_merge_flat = np.nanmean(sc_matrix_merge,axis=2)

plt.imshow(sc_matrix_merge_flat*1 ,cmap='PiYG', vmin=0.5,vmax=1.5)