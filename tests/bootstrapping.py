#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:01:34 2023

@author: marcnol
"""

import numpy as np

rng = np.random.default_rng()

from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

dist = norm(loc=.4, scale=.1)  # our "unknown" distribution

data = dist.rvs(size=10000, random_state=rng)
fig, ax = plt.subplots()

ax.hist(data, bins=25)

std_true = dist.std()      # the true value of the statistic

#%%
print(std_true)

std_sample = np.std(data)  # the sample statistic

print(std_sample)



data = (data,)  # samples must be in a sequence

res = bootstrap(data, np.mean, confidence_level=0.9,
                n_resamples=9999,
                batch = None,
                random_state=rng)

fig, ax = plt.subplots()

ax.hist(res.bootstrap_distribution, bins=25)

ax.set_title('Bootstrap Distribution')

ax.set_xlabel('statistic value')

ax.set_ylabel('frequency')

plt.show()

mean = np.mean(res.bootstrap_distribution)

error_mean = np.std(res.bootstrap_distribution)
print(f" mean: {mean} +- {error_mean}")

#%% single bin in a matrix

f = "/home/marcnol/Documents/tmp/"
file = f+"Trace_Trace_all_ROIs_filtered_exocrine_mask0_exp_ND_1_PDX1LR_Matrix_PWDscMatrix.npy"

import numpy as np

matrix_sc = np.load(file)
print(f"N of cells: {matrix_sc.shape[2]}")

x = matrix_sc[1,2,:]
x = x[~np.isnan(x)]

fig, ax = plt.subplots()

ax.hist(x, bins=25)
print(f" median from distribution: {np.nanmedian(x)} ")

def bootstrapping(x):
    data = (x,)  # samples must be in a sequence
    
    res = bootstrap(data, np.mean, confidence_level=0.9,
                    n_resamples=9999,
                    batch = None,
                    random_state=rng)
    
    return res.bootstrap_distribution

  
bs = bootstrapping(x)

fig, ax = plt.subplots()

ax.hist(bs, bins=25)

ax.set_title('Bootstrap Distribution')

ax.set_xlabel('statistic value')

ax.set_ylabel('frequency')

plt.show()

mean = np.nanmedian(bs)
error_mean = np.std(bs)
print(f" BS median: {mean} +- {error_mean}")
#%% all bins in a matrix

import numpy as np
rng = np.random.default_rng()

from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
from tqdm import trange

def bootstrapping(x):
    data = (x,)  # samples must be in a sequence
    
    res = bootstrap(data, np.mean, confidence_level=0.9,
                    n_resamples=9999,
                    batch = None,
                    random_state=rng)
    
    return res.bootstrap_distribution

def bootstraps_matrix(m):
    n_bins = m.shape[0]
    mean_bs = np.zeros((n_bins,n_bins))
    mean_error = np.zeros((n_bins,n_bins))
    print(f"n bins: {n_bins}")    
    
    for i in trange(n_bins):
        for j in range(i+1,n_bins):
            if i != j:
                # gets distribution and removes nans
                x = m[i,j,:]
                x = x[~np.isnan(x)]
                
                # bootstraps distribution
                bs = bootstrapping(x)
 
                # gets mean and std of mean
                mean_bs[i,j] = np.median(bs)
                mean_error[i,j] = np.std(bs)
                
                # symmetrizes matrix
                mean_bs[j,i] = mean_bs[i,j]
                mean_error[j,i] = np.std(bs)

    for i in range(n_bins):
        mean_bs[i,i], mean_error[i,i] = np.nan, np.nan
                
    return mean_bs, mean_error

f = "/home/marcnol/Documents/tmp/"
file = f+"Trace_Trace_all_ROIs_filtered_exocrine_mask0_exp_ND_1_PDX1LR_Matrix_PWDscMatrix.npy"
file = f+"Trace_Trace_all_ROIs_filtered_beta_mask0_exp_ND_1_PDX1LR_Matrix_PWDscMatrix.npy"

matrix_sc = np.load(file)
print(f"N of cells: {matrix_sc.shape[2]}")

mean_bs, mean_error = bootstraps_matrix(matrix_sc)

#%%

fig, ax = plt.subplots()
median_map = np.nanmean(matrix_sc,axis=2)
pos1=ax.imshow(median_map, cmap="RdBu", vmin = 0.15, vmax = .9*np.nanmax(median_map))
cbar = plt.colorbar(pos1, ax=ax, fraction=0.046, pad=0.04)
            
fig, ax = plt.subplots()
pos1=ax.imshow(mean_bs, cmap="RdBu", vmin = 0.15, vmax = .9*np.nanmax(mean_bs))
cbar = plt.colorbar(pos1, ax=ax, fraction=0.046, pad=0.04)

fig, ax = plt.subplots()
pos2=ax.imshow(mean_error, cmap="coolwarm", vmax = 1*np.nanmax(mean_error))
cbar = plt.colorbar(pos2, ax=ax, fraction=0.046, pad=0.04)

fig, ax = plt.subplots()
pos2=ax.imshow(mean_bs/mean_error, cmap="RdBu_r", vmax = 1*np.nanmax(mean_bs/mean_error))
cbar = plt.colorbar(pos2, ax=ax, fraction=0.046, pad=0.04)

