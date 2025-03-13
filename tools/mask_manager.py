#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# marcnol March 12 2025
#

import os
import uuid
import numpy as np
import h5py
from skimage.measure import regionprops_table
from scipy import ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

class MaskManager:
    """
    A class for managing 3D labeled image masks with efficient storage and retrieval.
    
    This class provides a solution for working with 3D labeled images where each unique
    integer value represents a different mask/object. It addresses the challenge of
    efficiently storing sparse 3D masks by storing only the coordinates of non-zero voxels.
    Additional properties of each mask (center of mass, diameter, voxel count, etc.) 
    are calculated and stored.
    
    Features:
    ---------
    - Convert 3D labeled images to compact HDF5 representation
    - Extract mask properties like center of mass, diameter, and voxel counts
    - Reconstruct original 3D images from the compressed representation
    - Add, remove, merge, or split individual masks
    - Efficient IO operations via HDF5 file format
    - imports masks from TIF/NPY labeled image 
    
    HDF5 File Structure:
    --------------------
    The HDF5 files created by this class have the following structure:
    
    / (root)
    ├── shape_x, shape_y, shape_z (attributes storing original image dimensions)
    └── masks/ (group)
        ├── mask_uuid_1/ (group)
        │   ├── label, center_of_mass_x/y/z, diameter, voxel_count, (attributes)
        │   └── coordinates (dataset: Nx3 array of XYZ coordinates)
        ├── mask_uuid_2/ (group)
        │   ├── label, center_of_mass_x/y/z, diameter, voxel_count, (attributes)
        │   └── coordinates (dataset: Nx3 array of XYZ coordinates)
        └── ... (more masks)
    
    Examples:
    ---------
    # Create a new manager with a specific shape
    >>> manager = MaskManager(shape=(100, 100, 100))
    
    # Extract masks from a labeled image
    >>> labeled_image = np.zeros((100, 100, 100), dtype=np.int32)
    >>> labeled_image[20:30, 20:30, 20:30] = 1
    >>> labeled_image[50:70, 50:70, 50:70] = 2
    >>> label_to_uuid = manager.extract_masks_from_labeled_image(labeled_image)
    
    # Save to file
    >>> manager.save_to_file("masks.hdf5")
    
    # Load from an existing file
    >>> manager = MaskManager(filename="masks.hdf5")
    
    # Get labeled image
    >>> reconstructed = manager.get_labeled_image()
    
    # Get information about all masks
    >>> mask_info = manager.get_mask_info()
    
    # Add a single mask
    >>> binary_mask = np.zeros((100, 100, 100), dtype=bool)
    >>> binary_mask[30:40, 30:40, 30:40] = True
    >>> mask_uuid = manager.add_mask_from_array(binary_mask, label=5)
    
    # Merge masks
    >>> mask_uuids = list(manager.masks.keys())[:2]  # First two masks
    >>> merged_uuid = manager.merge_masks(mask_uuids, new_label=10)
    
    # Split a mask using a labeled array
    >>> labels_array = np.zeros((100, 100, 100), dtype=np.int32)
    >>> mask_array = manager.get_mask_array(mask_uuid)
    >>> labels_array[mask_array] = np.random.randint(1, 4, size=np.sum(mask_array))
    >>> split_uuids = manager.split_mask(mask_uuid, labels_array)
    """
    
    def __init__(self, shape=None, filename=None, image = None):
        """
        Initialize a MaskManager object.
        
        Parameters:
        -----------
        shape : tuple, optional
            Shape of the 3D image (x, y, z). Required if creating a new manager.
        filename : str, optional
            Path to an existing HDF5 file to load. If provided, shape is ignored.
        """
        self.masks = {}  # Dictionary to store mask information
        
        if filename and os.path.exists(filename):
            # Determine file type from extension
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Load the image data based on file type
            if file_ext == '.hdf5':
                self.load_from_file_hdf5(filename)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}. Only .hdf5 is supported.")

        elif image is not None:
            # Print original datatype
            print(f"Original datatype/size: {image.dtype}/{image.size * image.itemsize / (1024*1024):.2f} MB")
            
            # Convert to uint32
            image = image.astype(np.uint32)
            print(f"Original datatype/size: {image.dtype}/{image.size * image.itemsize / (1024*1024):.2f} MB")
            
            self.load_from_image(image)

        elif shape:
            self.shape = shape
            self.labeled_image = np.zeros(shape, dtype=np.int32)
        else:
            raise ValueError("Either shape or a valid filename must be provided")
    
    def add_mask_from_array(self, mask_array, label, mask_uuid=None):
        """
        Add a mask from a binary or labeled array.
        
        Parameters:
        -----------
        mask_array : numpy.ndarray
            Binary array where True/non-zero values represent the mask.
        label : int
            Integer value to assign to this mask.
        mask_uuid : str, optional
            UUID for the mask. If not provided, a new one will be generated.
            
        Returns:
        --------
        str
            UUID of the added mask.
        """
        if mask_array.shape != self.shape:
            raise ValueError(f"Mask shape {mask_array.shape} doesn't match expected shape {self.shape}")
        
        # Create binary mask
        binary_mask = mask_array > 0
        
        # Generate UUID if not provided
        if mask_uuid is None:
            mask_uuid = str(uuid.uuid4())
            
        # Get coordinates of mask voxels
        mask_coords = np.where(binary_mask)
        x_coords, y_coords, z_coords = mask_coords
        
        # Skip if mask is empty
        if len(x_coords) == 0:
            return None
            
        # Store coordinates as Nx3 array
        coordinates = np.column_stack((x_coords, y_coords, z_coords)).astype(np.int32)
        
        # Calculate mask properties
        center_of_mass = ndimage.center_of_mass(binary_mask)
        
        # Calculate diameter (approximate)
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        diameter = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2 + (z_max - z_min)**2)
        
        # Count voxels
        voxel_count = len(x_coords)
    
        # Store mask information
        self.masks[mask_uuid] = {
            'label': int(label),
            'coordinates': coordinates,
            'center_of_mass': tuple(map(float, center_of_mass)),
            'diameter': float(diameter),
            'voxel_count': int(voxel_count),
        }
        
        # Update the labeled image with this mask
        for x, y, z in coordinates:
            self.labeled_image[x, y, z] = label
            
        return mask_uuid
    
    def add_mask_from_label(self, labeled_image, label, mask_uuid=None):
        """
        Extract and add a mask from a labeled image based on the provided label.
        
        Parameters:
        -----------
        labeled_image : numpy.ndarray
            3D array where each unique integer value represents a different mask.
        label : int
            Label value to extract from the labeled image.
        mask_uuid : str, optional
            UUID for the mask. If not provided, a new one will be generated.
            
        Returns:
        --------
        str
            UUID of the added mask.
        """
        mask = labeled_image == label
        return self.add_mask_from_array(mask, label, mask_uuid)

    def load_from_image(self, labeled_image, max_masks_per_chunk=2000):
        """ Load a 3D labeled image and extract masks efficiently in dynamically sized chunks. """

        if labeled_image.ndim != 3:
            if labeled_image.ndim == 2:
                labeled_image = labeled_image.reshape(1, *labeled_image.shape)  # Convert 2D to 3D
            else:
                raise ValueError(f"Invalid dimensions: {labeled_image.shape}. Expected a 3D array.")

        self.shape = labeled_image.shape
        print(f"Read image file with shape: {self.shape}")

        self.masks = {}
        self.labeled_image = np.zeros(self.shape, dtype=np.int32)

        unique_labels = np.unique(labeled_image)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background
        total_labels = len(unique_labels)

        if total_labels == 0:
            print("No masks found in the image.")
            return {}

        # Determine the optimal number of chunks
        num_chunks = max(1, total_labels // max_masks_per_chunk)
        chunk_size = max(1, total_labels // num_chunks)  # Adjusted chunk size
        print(f"Total masks: {total_labels}, Processing in {num_chunks} chunks (~{chunk_size} masks per chunk).")

        label_to_uuid = {}

        # Process image in chunks based on mask numbers
        for i in range(0, total_labels, chunk_size):
            label_subset = unique_labels[i : i + chunk_size]

            print(f"Processing chunk {i//chunk_size + 1}/{num_chunks}...")

            # Create a temporary labeled image with only these labels
            temp_image = np.isin(labeled_image, label_subset) * labeled_image

            # Compute properties in one batch
            properties = regionprops_table(
                temp_image,
                properties=('label', 'coords', 'centroid', 'equivalent_diameter_area')
            )

            # Store results
            for j in range(len(properties['label'])):
                label = properties['label'][j]
                coords = properties['coords'][j]  # (N, 3) array of (z, y, x)
                diameter = properties['equivalent_diameter_area'][j]

                mask_uuid = str(uuid.uuid4())

                center_of_mass = (
                    properties['centroid-2'][j],  # x
                    properties['centroid-1'][j],  # y
                    properties['centroid-0'][j]   # z
                )

                self.masks[mask_uuid] = {
                    'label': label,
                    'coordinates': coords[:, [2, 1, 0]],  # Convert (z, y, x) -> (x, y, z)
                    'center_of_mass': tuple(map(float, center_of_mass)),
                    'diameter': float(diameter),
                    'voxel_count': len(coords),
                }

                # Vectorized update to labeled image
                x, y, z = coords[:, 2], coords[:, 1], coords[:, 0]
                self.labeled_image[z, y, x] = label

                label_to_uuid[label] = mask_uuid

        print(f"Successfully processed {len(label_to_uuid)} masks from image")
        return label_to_uuid
    ''' 
    def load_from_image(self, labeled_image):
        """
        Load a 3D labeled image from a TIF or NPY file and extract all masks into the manager.

        Parameters:
        -----------
        filename : str
            Path to the TIF or NPY file containing a 3D labeled image.

        Returns:
        --------
        dict
            Dictionary mapping labels to UUIDs.
        """

        # Handle different dimensions
        if labeled_image.ndim != 3:
            if labeled_image.ndim == 2:
                labeled_image = labeled_image.reshape(1, *labeled_image.shape)  # Convert 2D to 3D
            else:
                raise ValueError(f"Invalid dimensions: {labeled_image.shape}. Expected a 3D array.")

        # Store image shape
        self.shape = labeled_image.shape
        print(f"Read image file with shape: {self.shape}")
        
        # Reset mask storage
        self.masks = {}
        self.labeled_image = np.zeros(self.shape, dtype=np.int32)

        # Extract region properties efficiently
        print(f"Extracting properties from {np.max(labeled_image)} masks...")
        properties = regionprops_table(
            labeled_image,
            properties=('label', 'coords', 'centroid','equivalent_diameter_area')
        )

        # Dictionary to store label-to-UUID mapping
        label_to_uuid = {}

        print(f"Processing {len(properties['label'])} masks...")

        for i in tqdm(range(len(properties['label'])), desc="Extracting masks"):
            label = properties['label'][i]
            coords = properties['coords'][i]  # (N, 3) array of (z, y, x) voxel coordinates
            diameter = properties['equivalent_diameter_area'][i]
            
            # Generate UUID for this mask
            mask_uuid = str(uuid.uuid4())

            # Convert center of mass to (x, y, z) format
            center_of_mass = (
                properties['centroid-2'][i],  # x
                properties['centroid-1'][i],  # y
                properties['centroid-0'][i]   # z
            )

            # Store mask metadata
            self.masks[mask_uuid] = {
                'label': label,
                'coordinates': coords[:, [2, 1, 0]],  # Convert (z, y, x) -> (x, y, z)
                'center_of_mass': tuple(map(float, center_of_mass)),
                'diameter': float(diameter),
                'voxel_count': len(coords),
            }

            # Update labeled image efficiently
            x, y, z = coords[:, 2], coords[:, 1], coords[:, 0]
            self.labeled_image[z, y, x] = label  # Use numpy vectorized assignment

            # Store label-to-UUID mapping
            label_to_uuid[label] = mask_uuid

        print(f"Successfully processed {len(label_to_uuid)} masks from image")
        return label_to_uuid
    '''

    
    def extract_masks_from_labeled_image(self, labeled_image, background=0):
        """
        Extract all masks from a labeled image and add them to the manager.
        
        Parameters:
        -----------
        labeled_image : numpy.ndarray
            3D array where each unique integer value represents a different mask.
        background : int, optional
            Value representing the background (to be ignored).
            
        Returns:
        --------
        dict
            Dictionary mapping labels to UUIDs.
        """
        if labeled_image.shape != self.shape:
            raise ValueError(f"Image shape {labeled_image.shape} doesn't match expected shape {self.shape}")
        
        # Reset current state
        self.masks = {}
        self.labeled_image = np.zeros(self.shape, dtype=np.int32)
        
        # Find unique labels (excluding background)
        unique_labels = np.unique(labeled_image)
        unique_labels = unique_labels[unique_labels != background]
        
        # Dictionary to track label to UUID mapping
        label_to_uuid = {}
        
        # Process each label
        for label in unique_labels:
            uuid = self.add_mask_from_label(labeled_image, label)
            label_to_uuid[int(label)] = uuid
            
        return label_to_uuid
    
    def get_mask_array(self, mask_uuid):
        """
        Generate a binary mask array for the specified mask UUID.
        
        Parameters:
        -----------
        mask_uuid : str
            UUID of the mask to retrieve.
            
        Returns:
        --------
        numpy.ndarray
            Binary array where True values represent the mask.
        """
        if mask_uuid not in self.masks:
            raise KeyError(f"Mask with UUID {mask_uuid} not found")
            
        mask_info = self.masks[mask_uuid]
        mask_array = np.zeros(self.shape, dtype=bool)
        
        coordinates = mask_info['coordinates']
        for x, y, z in coordinates:
            mask_array[x, y, z] = True
            
        return mask_array
    
    def remove_mask(self, mask_uuid):
        """
        Remove a mask from the manager.
        
        Parameters:
        -----------
        mask_uuid : str
            UUID of the mask to remove.
            
        Returns:
        --------
        bool
            True if the mask was successfully removed, False otherwise.
        """
        if mask_uuid not in self.masks:
            return False
            
        mask_info = self.masks[mask_uuid]
        label = mask_info['label']
        
        # Update the labeled image by removing this mask
        coordinates = mask_info['coordinates']
        for x, y, z in coordinates:
            if self.labeled_image[x, y, z] == label:
                self.labeled_image[x, y, z] = 0
                
        # Remove from masks dictionary
        del self.masks[mask_uuid]
        return True
    
    def get_labeled_image(self):
        """
        Get the current 3D labeled image.
        
        Returns:
        --------
        numpy.ndarray
            3D array where each unique integer value represents a different mask.
        """
        # For efficiency, we maintain the labeled image as we add/remove masks,
        # so we can just return it directly
        return self.labeled_image
    
    def reconstruct_labeled_image(self):
        """
        Reconstruct the labeled image from the stored mask information.
        This is useful for validation or if you suspect the internal labeled image
        might be out of sync with the mask data.
        
        Returns:
        --------
        numpy.ndarray
            Reconstructed 3D labeled image.
        """
        reconstructed = np.zeros(self.shape, dtype=np.int32)
        
        for mask_uuid, mask_info in self.masks.items():
            label = mask_info['label']
            coordinates = mask_info['coordinates']
            
            for x, y, z in coordinates:
                try:
                    reconstructed[x, y, z] = label
                except IndexError:
                    print(f"Warning: Index ({x}, {y}, {z}) out of bounds for shape {self.shape}")
                    continue
                    
        return reconstructed
    
    def save_to_file(self, filename):
        """
        Save the mask data to an HDF5 file.
        
        Parameters:
        -----------
        filename : str
            Output HDF5 filename.
            
        Returns:
        --------
        bool
            True if saved successfully.
        """
        with h5py.File(filename, 'w') as f:
            # Store original image shape
            f.attrs['shape_x'] = int(self.shape[0])
            f.attrs['shape_y'] = int(self.shape[1])
            f.attrs['shape_z'] = int(self.shape[2])
            
            # Create a group for masks
            masks_group = f.create_group('masks')
            
            # Store each mask
            for mask_uuid, mask_info in self.masks.items():
                # Create a group for this mask
                mask_group = masks_group.create_group(mask_uuid)
                
                # Store mask properties
                mask_group.attrs['label'] = int(mask_info['label'])
                
                com = mask_info['center_of_mass']
                mask_group.attrs['center_of_mass_x'] = float(com[0])
                mask_group.attrs['center_of_mass_y'] = float(com[1])
                mask_group.attrs['center_of_mass_z'] = float(com[2])
                
                mask_group.attrs['diameter'] = float(mask_info['diameter'])
                mask_group.attrs['voxel_count'] = int(mask_info['voxel_count'])
                
                # Store XYZ coordinates directly
                coordinates = np.array(mask_info['coordinates'], dtype=np.int32)
                mask_group.create_dataset('coordinates', data=coordinates)
                
        print(f"File {filename} saved")
        return True
    
    def load_from_file_hdf5(self, filename):
        """
        Load mask data from an HDF5 file.
        
        Parameters:
        -----------
        filename : str
            Input HDF5 filename.
            
        Returns:
        --------
        bool
            True if loaded successfully.
        """
        with h5py.File(filename, 'r') as f:
            # Get original image shape
            shape_x = int(f.attrs['shape_x'])
            shape_y = int(f.attrs['shape_y'])
            shape_z = int(f.attrs['shape_z'])
            self.shape = (shape_x, shape_y, shape_z)
            
            print(f"Apparent image shape {self.shape}")
            # Initialize empty labeled image
            self.labeled_image = np.zeros(self.shape, dtype=np.int32)
            
            # Clear existing masks
            self.masks = {}
            
            # Process each mask
            masks_group = f['masks']
            for mask_uuid in masks_group:
                mask_group = masks_group[mask_uuid]
                
                # Get mask properties
                label = int(mask_group.attrs['label'])
                
                center_of_mass = (
                    float(mask_group.attrs['center_of_mass_x']),
                    float(mask_group.attrs['center_of_mass_y']),
                    float(mask_group.attrs['center_of_mass_z'])
                )
                
                # Get coordinates
                coordinates = np.array(mask_group['coordinates'][:], dtype=np.int32)
                
                # Store mask information
                self.masks[mask_uuid] = {
                    'label': label,
                    'coordinates': coordinates,
                    'center_of_mass': center_of_mass,
                    'diameter': float(mask_group.attrs['diameter']),
                    'voxel_count': int(mask_group.attrs['voxel_count']),
                }
                
                # Reconstruct this mask in the labeled image
                for z, y, x in coordinates:
                    try:
                        self.labeled_image[x, y, z] = label
                    except IndexError:
                        print(f"Warning: Index out of bounds while reconstructing mask {label}|({x},{y},{z})|{mask_uuid}")
                        continue
                        
        return True
    
    def get_mask_info(self, mask_uuid=None):
        """
        Get information about masks.
        
        Parameters:
        -----------
        mask_uuid : str, optional
            UUID of a specific mask to get information for.
            If None, returns information for all masks.
            
        Returns:
        --------
        dict
            Dictionary containing mask information.
        """
        if mask_uuid:
            if mask_uuid not in self.masks:
                raise KeyError(f"Mask with UUID {mask_uuid} not found")
            
            # Return a copy without the coordinates array (which could be very large)
            info = self.masks[mask_uuid].copy()
            info['coordinates'] = f"<{len(info['coordinates'])} voxels>"
            return info
        else:
            # Return summary information for all masks
            summary = {}
            for uuid, mask_info in self.masks.items():
                summary[uuid] = {
                    'label': mask_info['label'],
                    'center_of_mass': mask_info['center_of_mass'],
                    'diameter': mask_info['diameter'],
                    'voxel_count': mask_info['voxel_count'],
                }
            return summary
            
    def merge_masks(self, mask_uuids, new_label=None):
        """
        Merge multiple masks into a single mask.
        
        Parameters:
        -----------
        mask_uuids : list
            List of UUIDs of masks to merge.
        new_label : int, optional
            Label for the merged mask. If not provided, uses the label of the first mask.
            
        Returns:
        --------
        str
            UUID of the merged mask.
        """
        if not mask_uuids:
            raise ValueError("No masks provided for merging")
            
        if not all(uuid in self.masks for uuid in mask_uuids):
            missing = [uuid for uuid in mask_uuids if uuid not in self.masks]
            raise KeyError(f"Masks not found: {missing}")
            
        # Create a binary mask for the merged result
        merged_mask = np.zeros(self.shape, dtype=bool)
        
        # Use label from first mask if not specified
        if new_label is None:
            new_label = self.masks[mask_uuids[0]]['label']
            
        # Combine all masks
        for mask_uuid in mask_uuids:
            mask_info = self.masks[mask_uuid]
            coordinates = mask_info['coordinates']
            for x, y, z in coordinates:
                merged_mask[x, y, z] = True
                
        # Remove original masks
        for mask_uuid in mask_uuids:
            self.remove_mask(mask_uuid)
            
        # Add the merged mask
        return self.add_mask_from_array(merged_mask, new_label)
            
    def split_mask(self, mask_uuid, labels_array):
        """
        Split a mask into multiple masks based on a labeled array.
        
        Parameters:
        -----------
        mask_uuid : str
            UUID of the mask to split.
        labels_array : numpy.ndarray
            Array of the same size as the image with different label values
            for different regions within the mask.
            
        Returns:
        --------
        dict
            Dictionary mapping new labels to new mask UUIDs.
        """
        if mask_uuid not in self.masks:
            raise KeyError(f"Mask with UUID {mask_uuid} not found")
            
        if labels_array.shape != self.shape:
            raise ValueError(f"Labels array shape {labels_array.shape} doesn't match expected shape {self.shape}")
            
        # Get the original mask
        original_mask = self.get_mask_array(mask_uuid)
        
        # Remove the original mask
        original_label = self.masks[mask_uuid]['label']
        self.remove_mask(mask_uuid)
        
        # Get unique labels within the mask area
        masked_labels = labels_array[original_mask]
        unique_labels = np.unique(masked_labels)
        unique_labels = unique_labels[unique_labels > 0]  # Ignore background
        
        # Dictionary to track new UUIDs
        new_mask_uuids = {}
        
        # Create new masks for each label
        for label in unique_labels:
            # Create a binary mask for this label
            sub_mask = np.zeros(self.shape, dtype=bool)
            sub_mask[original_mask & (labels_array == label)] = True
            
            # Add the new mask with the original label + new label as a decimal
            new_label = original_label * 100 + label  # e.g., if original is 5, new becomes 501, 502, etc.
            new_uuid = self.add_mask_from_array(sub_mask, new_label)
            new_mask_uuids[int(label)] = new_uuid
            
        return new_mask_uuids



    def plot_xy_projection(self, figsize=(10, 8), cmap=None, title="XY Projection of All Masks", 
                        alpha=1.0, random_colors=True, background_color='black', dpi=100, save_path=None):
        """
        Plot an XY projection of all masks in the MaskManager.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches.
        cmap : matplotlib colormap, optional
            Colormap for the masks. If None and random_colors is False, 'viridis' is used.
        title : str, optional
            Title for the plot.
        alpha : float, optional
            Transparency level of the masks (0.0 to 1.0).
        random_colors : bool, optional
            If True, assign random colors to each mask. If False, use the label values with cmap.
        background_color : str, optional
            Color for the background (areas with no masks).
        dpi : int, optional
            Resolution of the output figure.
        save_path : str, optional
            If provided, save the figure to this path.
            
        Returns:
        --------
        tuple
            (fig, ax) - The figure and axis objects.
        """
        # Get the labeled image from the manager
        labeled_image = self.get_labeled_image()
        
        # Create an XY projection (maximum intensity projection)
        projection = np.max(labeled_image, axis=0)
        
        # Get unique labels (excluding background)
        unique_labels = np.unique(projection)
        unique_labels = unique_labels[unique_labels > 0]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        if random_colors:
            # Create a colormap with random colors for each mask
            # First color is background, rest are for masks
            n_labels = len(unique_labels) + 1
            
            # Generate random colors for each label
            # Convert background color to RGB
            if background_color == 'black':
                bg_color = [0, 0, 0]
            elif background_color == 'white':
                bg_color = [1, 1, 1]
            else:
                # Try to parse the background color
                try:
                    from matplotlib.colors import to_rgb
                    bg_color = list(to_rgb(background_color))
                except:
                    bg_color = [0, 0, 0]  # Default to black if parsing fails
            
            # Initialize colors with background color
            colors = [bg_color]
            
            # Generate random colors for each label
            for _ in range(n_labels - 1):
                colors.append([random.random(), random.random(), random.random()])
            
            # Create a custom colormap
            custom_cmap = ListedColormap(colors)
            
            # Plot the projection with random colors
            im = ax.imshow(projection, cmap=custom_cmap, interpolation='nearest', alpha=alpha)
        else:
            # Use the provided colormap or default to viridis
            if cmap is None:
                cmap = plt.cm.viridis
            
            # Plot the projection
            im = ax.imshow(projection, cmap=cmap, interpolation='nearest', alpha=alpha)
        
        # Add a colorbar
        plt.colorbar(im, ax=ax, label='Mask Label')
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel('X Dimension')
        ax.set_ylabel('Y Dimension')
        
        # Add information about the number of masks
        ax.text(0.02, 0.98, f"Number of masks: {len(unique_labels)}", 
                transform=ax.transAxes, color='white', fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='gray', alpha=0.5))
        
        # Tight layout
        plt.tight_layout()
        
        # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
        return fig, ax

def example():
    """
    Demonstrates the usage of the MaskManager class with a simple example.
    
    This function:
    1. Creates a sample 3D labeled image
    2. Extracts masks using MaskManager
    3. Saves masks to an HDF5 file
    4. Loads masks from the file
    5. Verifies correct reconstruction
    6. Demonstrates mask merging
    
    Returns:
    --------
    tuple
        A tuple containing (original_manager, loaded_manager)
    """
    # Create sample data
    shape = (100, 100, 100)
    labeled_image = np.zeros(shape, dtype=np.int32)
    
    # Add some example masks
    labeled_image[20:30, 20:30, 20:30] = 1
    labeled_image[50:70, 50:70, 50:70] = 2
    
    # Create mask manager
    manager = MaskManager(shape=shape)
    
    # Extract masks from labeled image
    label_to_uuid = manager.extract_masks_from_labeled_image(labeled_image)
    print(f"Extracted masks: {label_to_uuid}")
    
    # Get information about a specific mask
    mask1_uuid = label_to_uuid[1]
    mask1_info = manager.get_mask_info(mask1_uuid)
    print(f"\nMask 1 info: {mask1_info}")
    
    # Save to file
    filename = "mask_manager_example.hdf5"
    manager.save_to_file(filename)
    print(f"\nSaved to {filename}")
    
    # Load from file
    new_manager = MaskManager(filename=filename)
    print(f"\nLoaded from file. Number of masks: {len(new_manager.masks)}")
    
    # Verify reconstruction
    original = manager.get_labeled_image()
    loaded = new_manager.get_labeled_image()
    match = np.array_equal(original, loaded)
    print(f"Reconstruction match: {match}")
    
    # Merge masks
    mask_uuids = list(new_manager.masks.keys())
    if len(mask_uuids) >= 2:
        merged_uuid = new_manager.merge_masks(mask_uuids[:2], new_label=3)
        print(f"\nMerged masks. New UUID: {merged_uuid}")
        print(f"Number of masks after merging: {len(new_manager.masks)}")
    
    return manager, new_manager

if __name__ == "__main__":
    example()