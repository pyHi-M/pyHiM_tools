#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:40:07 2024

@author: legall
"""

import cv2 as cv
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp
import tifffile
from skimage.filters import difference_of_gaussians
from scipy.ndimage import map_coordinates

import time

tic = time.time()

# Get an optical flow model. 
model_name = 'fastflownet'  # [other models]
pretrained_weight = 'sintel'  # [chairs, things, sintel, kitti]
model = ptlflow.get_model(model_name, pretrained_ckpt=pretrained_weight)

# Load the images
filename = "/mnt/grey/DATA/users/Antoine/Git/RAFT/Test_registration_RTs/Experiment_76_RT1_RT17.tif"
stack = tifffile.imread(filename)[:, 0:512, 0:512]

# Apply Difference of Gaussians filter to each frame to keep high spatial frequencies
# filtered_frames = [difference_of_gaussians(frame, low_sigma=0.1, high_sigma=5) for frame in stack]
# stack = np.array(filtered_frames)

# Extract a pair of frames from the stack
frames_num = [0, 1]
frames = stack[frames_num]

# Check the type and shape of the frames
print(f"Frame dtype: {frames.dtype}, shape: {frames.shape}")

# Convert frames to uint8 if necessary (for compatibility with optical flow models)
if frames.dtype != np.uint8:
    frames_uint8 = []
    for frame in frames:
        frame = frame.astype(np.float32)
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
        frame = (frame * 255).astype(np.uint8)
        frames_uint8.append(frame)
    frames = np.array(frames_uint8)


# Convert frames to RGB format if they are single channel
if len(frames.shape) == 3 and frames.shape[1] == frames.shape[2]:
    frames_rgb = [cv.cvtColor(frame, cv.COLOR_GRAY2BGR) for frame in frames]
else:
    frames_rgb = frames

# A helper to manage inputs and outputs of the model
io_adapter = IOAdapter(model, frames_rgb[0].shape[:2])

# inputs is a dict {'images': torch.Tensor}
inputs = io_adapter.prepare_inputs(frames_rgb)

# Forward the inputs through the model
predictions = model(inputs)

# The output is a dict with possibly several keys,
# but it should always store the optical flow prediction in a key called 'flows'.
flows = predictions['flows']

# flows will be a 5D tensor BNCHW.
# This example should print a shape (1, 1, 2, H, W).
print(flows.shape)

# Create an RGB representation of the flow to show it on the screen
flow_rgb = flow_utils.flow_to_rgb(flows)
# Make it a numpy array with HWC shape
flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
flow_rgb_npy = flow_rgb.detach().cpu().numpy()

# Function to predict im1 based on flow
def predict_im1(labeled_im0, flow): 
    '''
    Transforms labeled_im0 image using the flow vector field.

    Parameters
    ----------
    labeled_im0 : 2D image of shape (H, W) or anything else
    flow : Flow field of shape (2, H, W)

    Returns
    -------
    image0_warped : warped image
    '''        
    nr, nc = labeled_im0.shape
    
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    
    # Get flow vectors
    v = flow[1, :, :]
    u = flow[0, :, :]
    
    # Coordinates with flow applied
    new_row_coords = row_coords - v
    new_col_coords = col_coords - u
    
    # Clip coordinates to be within image bounds
    new_row_coords = np.clip(new_row_coords, 0, nr - 1)
    new_col_coords = np.clip(new_col_coords, 0, nc - 1)
    
    # Warp image using map_coordinates for better interpolation
    image0_warped = map_coordinates(labeled_im0, [new_row_coords, new_col_coords], order=1, mode='nearest')
    
    return image0_warped

# Extract the flow field
flow_field = flows[0, 0].detach().cpu().numpy()

# Use the flow field to align the images
aligned_image = predict_im1(frames[0], flow_field)

# Visualization
nr, nc = frames[0].shape
RGB_original = np.zeros((nr, nc, 3))
RGB_original[..., 0] = frames[0]  # frame 0 in red
RGB_original[..., 1] = frames[1]  # frame 1 in green

RGB_aligned = np.zeros((nr, nc, 3))
RGB_aligned[..., 0] = aligned_image  # aligned frame 0 in red
RGB_aligned[..., 1] = frames[1]      # frame 1 in green

# Normalize RGB data to [0, 1] range for display
RGB_original = RGB_original / np.max(RGB_original)
RGB_aligned = RGB_aligned / np.max(RGB_aligned)

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, sharey=True)
ax0.imshow(RGB_original)
ax0.set_title("Im0=Red, Im1=Green : " + str(frames_num))
ax0.set_axis_off()

ax1.imshow(flow_rgb_npy, interpolation='None')
ax1.set_title(model_name + "  +  " + pretrained_weight)
ax1.set_axis_off()

ax2.imshow(RGB_aligned)
ax2.set_title("Aligned Im0 (Red) over Im1 (Green)")
ax2.set_axis_off()

fig.tight_layout()
plt.show()

print('Elapsed time = ', time.time()-tic)