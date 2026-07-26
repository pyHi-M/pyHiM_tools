# -*- coding: utf-8 -*-
"""
@authors: JB. Fiche
Created on Mon January 10, 2022
------------------------------------------------------------------------------------------------------------------------
Purpose
-------
This script aligns a target DAPI image stack onto a reference DAPI image stack. It was designed for experiments in which
the same field of view is imaged before and after disassembly and reassembly of the fluidics chamber.

A typical application is the alignment of DNA-HiM and RNA-imaging experiments. The resulting transformation can be used
to express signals from the target experiment in the coordinate system of the reference experiment, allowing DNA traces,
segmented cells, and RNA measurements to be compared.

The reference stack remains fixed. The target stack is rotated, translated, and optionally resized to match the reference stack.

Registration procedure
----------------------
1. Locate exactly one reference TIFF file and one target TIFF file using the folders and filename patterns specified in the
   YAML configuration.
2. Extract the selected channel from each interleaved multichannel stack.
3. Calculate a maximum-intensity projection along Z.
4. Correct large-scale illumination variations and standardize intensities.
5. Estimate an in-plane rotation and XY translation by maximizing normalized cross-correlation.
6. Optionally estimate independent X and Y zoom factors.
7. Optionally estimate a Z displacement from orthogonal projections.
8. Save the estimated parameters and registration diagnostic images.

Inputs
------
A YAML configuration file specifying:

- the reference and target folders;
- filename patterns identifying the TIFF stacks;
- the total number of interleaved channels in each stack;
- the zero-based DAPI channel index;
- registration search ranges;
- the destination folder;
- whether existing results may be overwritten.

Usage
-----
Run from a terminal:

    python Image_registration.py -F /path/to/Config_registration.yml

For pycharm, the arguments are defined in Configurations/script parameters. If the configuration file is moved to
another location, the arguments must be updated.
------------------------------------------------------------------------------------------------------------------------
"""
import time

from tifffile import imread, TiffWriter, TiffFile
from scipy.signal import correlate, fftconvolve
from skimage.registration import phase_cross_correlation
from scipy.optimize import minimize, differential_evolution
from scipy.ndimage import rotate, gaussian_filter, shift, zoom
from scipy.stats import pearsonr
from tqdm import tqdm
from functools import wraps
from time import time
from glob import glob
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import json
import logging
import shutil
import yaml

# matplotlib.use('TkAgg')
matplotlib.use('Agg')
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


# Defines the decorator function for the log
def log(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        t0 = time()
        result = func(*args, **kwargs)
        t1 = time()
        print(f'function : {func.__name__} - execution time = {t1 - t0}s')
        return result
    return wrap


# def plot_images(im0, im):
#     plt.figure(figsize=(16, 10))
#     plt.imshow(im0, cmap='Blues')
#     plt.imshow(im, cmap='Reds', alpha=0.5)
#     plt.show()


class ImageRegistration:

    def __init__(self, config_file):

        config_file = Path(config_file).expanduser()

        if not config_file.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}"
            )

        with config_file.open("r", encoding="utf-8") as configuration_file:
            parameters = yaml.safe_load(configuration_file)

        registration_parameters = parameters['registration_parameters']
        self.data_tag: str = parameters['data_tag']
        self.downsizing_power: int = registration_parameters['downsizing_power']
        self.gaussian_filter_size: int = registration_parameters['gaussian_filter_size']
        self.angles_range: tuple = registration_parameters['angles_range']
        self.zoom_range: tuple = registration_parameters['zoom_range']
        self.apply_zoom: bool = registration_parameters['apply_zoom']
        self.crude_search_angle: int = np.ceil(np.sqrt((self.angles_range[1] - self.angles_range[0]) / 2))
        self.verbose: bool = registration_parameters['verbose']

        self.path_image_ref: str = parameters['reference_DAPI_folder']
        self.template_ref_file: str = parameters['template_reference_DAPI_file']
        self.ref_total_number_channels: int = parameters['number_channel_reference_stack']
        self.ref_channel: int = parameters['reference_channel']

        self.path_image_to_align: str = parameters['target_DAPI_folder']
        self.template_target_file: str = parameters['template_target_DAPI_file']
        self.align_total_number_channels: int = parameters['number_channel_target_stack']
        self.align_channel: int = parameters['target_channel']

        self.saving_path: str = os.path.join(parameters['dest_root_folder'],
                                             parameters['dest_analysis_folder'])
        self.overwrite_existing_file: bool = parameters['overwrite_existing_file']
        previous_analysis = self.create_saving_folder()

        if previous_analysis:
            self.reload_analysis = True
            self.load_registration_dic(previous_analysis[0])
        else:
            self.reload_analysis = False
            self.registration_dic: dict = {"shift": [0, 0],
                                           "rotation": 0,
                                           "zoom": [1, 1],
                                           "z_shift": 0}

    @staticmethod
    def parse_arguments():
        """ Parse the argument to retrieve the path to the configuration file.

        @return: parsed arguments.
        """
        parser = ArgumentParser(
            description=(
                "Register a target DAPI image stack onto a reference "
                "DAPI image stack."
            )
        )
        parser.add_argument(
            "-F",
            "--config-file",
            dest="config_file",
            type=Path,
            required=True,
            help="Path to the YAML configuration file.",
        )

        return parser.parse_args()

    def create_saving_folder(self):
        """ Check if the folder already exist. If it does, depending on the "overwrite_existing_file" option, the
        previous is removed or the analysis will be stopped.

        @return: (bool) True, a previous analysis was found.
        """
        # Check if the folder already exist. and create the saving folder
        if os.path.isdir(self.saving_path):
            if self.overwrite_existing_file:
                for filename in os.listdir(self.saving_path):
                    if self.data_tag in filename:
                        file_path = os.path.join(self.saving_path, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.remove(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            print(f"Failed to delete {file_path}. Reason: {e}")
                return None
            else:
                print(f'The saving path already exists for {self.data_tag}.')
                previous_registration_file = glob(os.path.join(self.saving_path,
                                                               f"{self.data_tag}_registration_parameters.json"))
                if len(previous_registration_file) == 1:
                    return previous_registration_file
                else:
                    return None
        else:
            os.mkdir(self.saving_path)
            return None

    @staticmethod
    def load_stack(path, total_number_channels, selected_channel):
        """ Load one channel from an interleaved multichannel TIFF stack.
        The TIFF is assumed to contain planes in the following order: Z0/channel 0, Z0/channel 1, ..., Z1/channel 0, Z1/channel 1, ...

        @param path: path to the selected stack
        @param total_number_channels: (int) total number of channels composing the stack
        @param selected_channel: (int) indicate the channel to extract
        @return: a stack of images corresponding to the selected channel. Returns ``None`` when zero or multiple files are found.
        """
        if len(path) == 1:
            im = imread(path[0])
            im = im[np.arange(start=selected_channel, stop=im.shape[0], step=total_number_channels, dtype=int), :, :]
            return im
        elif len(path) > 1:
            logger.warning('Load_stack was aborted since more than one path was detected.')
            return None
        else:
            logger.warning('Load_stack was aborted, no file was found.')
            return None

    @staticmethod
    def mip(stack):
        """ Calculate the maximum intensity projection of an input stack

        @param stack: input stack of images corresponding to a single acquisition channel
        @return: the mip as float
        """
        mip = np.max(stack, axis=0)
        mip = mip.astype(np.float_)
        return mip

    def bin_stack(self, stack):
        """ Apply binning to all images of a stack

        @param stack: input stack of images
        @return: binned stack
        """
        lz, lx, ly = stack.shape
        binning_factor = 2 ** self.downsizing_power
        binned_stack = np.zeros((lz, lx//binning_factor, ly//binning_factor))

        for frame in tqdm(range(lz)):
            binned_stack[frame, :, :] = self.bin_image(stack[frame, :, :])

        return binned_stack

    @staticmethod
    def convert_to_8bit(im):
        """ Convert image to 8 bit in order to rescale intensity

        @param im: input 2d image
        @return: image_normalized (np array) the 8bit normalized image
        """
        v_min, v_max = np.percentile(im, (0.1, 99.9))  # Ignore extreme outliers (1rst and 99th percentiles)
        image_clipped = np.clip(im, v_min, v_max)  # Clip to [v_min, v_max]
        image_normalized = ((image_clipped - v_min) / (v_max - v_min) * 255).astype(np.uint8)
        return image_normalized

    def bin_image(self, im):
        """ Bin (down-sizing) the input image in order to keep the essential structural information while saving time
        later during the alignment process. Indeed, the calculation for the correlation is much faster when working on
        small images.

        @param im: input 2d image
        @return: the binned image
        """
        # Bin the image according to the downsizing power indicated as parameter
        resize_shape = (int(im.shape[0] / 2 ** self.downsizing_power), int(im.shape[1] / 2 ** self.downsizing_power))

        shape = (resize_shape[0], im.shape[0] // resize_shape[0],
                 resize_shape[1], im.shape[1] // resize_shape[1])
        im_bin = im.reshape(shape).mean(-1).mean(1)

        return im_bin

    def process_image(self, im, downsize=False):
        """ The image is divided by a Gaussian-smoothed copy to reduce gradual illumination variation. The result
         is then converted to zero mean and unit standard deviation. This processed image is used only to estimate
         registration parameters; it is not saved as the registered output.

        @param im: input 2d image
        @param downsize: indicate whether downsizing should be applied
        @return: mip : maximum intensity projection of the original stack im
                 im_standardized : return the standardized mip after binning
        """
        # define the size of the gaussian kernel
        if downsize:
            filter_size = self.gaussian_filter_size / (2 * self.downsizing_power)
        else:
            filter_size = self.gaussian_filter_size

        # Correct for illumination inhomogeneity using a gaussian filter
        if np.min(im) > 0:
            im_processed = np.divide(im, gaussian_filter(im, filter_size))
        else:
            im = im + 10**-6
            im_processed = np.divide(im, gaussian_filter(im, filter_size))

        # Standardized the image
        im_standardized = (im_processed - np.mean(im_processed)) / np.std(im_processed)

        return im_standardized

    @staticmethod
    def im_rotate(im, angle_value):
        """ Rotate an image and automatically crop the valid region that will exclude padding artifacts whatever the
        value of the rotation angle. This is important since correlation / RMS error will vary depending on the size of
        the images.

        @param im: input 2D image
        @param angle_value: value of the rotation
        @return: im_cropped, the cropped central part of the rotated image
        """
        # apply rotation by angle_value while keeping the same shape of the image
        im_rotated = rotate(im, angle_value, axes=(1, 0), reshape=False, order=1, mode='constant', prefilter=True)

        # select the central part of the rotated image (the one that will never be modified by padding whatever the
        # value of theta)
        im_size_x, im_size_y = im.shape
        crop_size_x = int(np.ceil((im_size_x - im_size_x / np.sqrt(2)) / 2))
        crop_size_y = int(np.ceil((im_size_y - im_size_y / np.sqrt(2)) / 2))
        im_cropped = im_rotated[crop_size_x:im_size_x - crop_size_x, crop_size_y:im_size_y - crop_size_y]

        # d_shift = np.ceil((im_size - im_size / np.sqrt(2)) / 2)
        # d_shift = d_shift.astype('int')
        # im_rotated = im_rotated[d_shift:im_size - d_shift, d_shift:im_size - d_shift]

        return im_cropped

    @staticmethod
    def crop_center(im, target_shape):
        """
        Crop the central region of `im` to match `target_shape`.

        @param im: 2D input image (numpy array)
        @param target_shape: (height, width) tuple for output size
        @return: Cropped image of shape `target_shape`, centered
        """
        h, w = im.shape
        th, tw = target_shape

        assert th <= h and tw <= w, "Target shape must be smaller than or equal to input shape."

        top = (h - th) // 2
        left = (w - tw) // 2
        return im[top:top + th, left:left + tw]

    @staticmethod
    def cross_correlation_fft(im1, im2):
        im1 = (im1 - np.mean(im1)) / (np.std(im1) + 1e-8)
        im2 = (im2 - np.mean(im2)) / (np.std(im2) + 1e-8)
        return fftconvolve(im1, im2[::-1, ::-1], mode='same')

    def angle_search(self, angles, n_angles, roi, roi_shape, im_ref):
        """ For each angle, perform a rotation of an ROI selected on the image we want to register. Correlate it to the
        reference image.

        @param angles: (ndarray) containing all the values to test
        @param n_angles: (int) number of angles to test
        @param roi: (2d ndarray) central roi selected on the 2d image we need to register
        @param roi_shape: (2d tuple) size of roi
        @param im_ref: (2d ndarray) reference image for the registration
        @return: correlation_max : (1d ndarray) maximum value for the correlation for each value of angles
                 max_correlation_2d_pos : (2d ndarray) x,y position of the pixel with the highest correlation value. It
                 will be used later to indicated by how much the image to register should be shifted with respect to the
                 reference.
        """
        correlation_max = np.zeros((n_angles,))
        correlation_im = np.zeros((n_angles, im_ref.shape[0], im_ref.shape[1]))
        max_correlation_2d_pos = np.zeros((n_angles, 2))

        # For each angle, rotate the roi and calculate the correlated image with the reference
        for n, angle in enumerate(angles):
            roi_rotated = self.im_rotate(roi, angle)
            corr = self.cross_correlation_fft(im_ref, roi_rotated)
            # corr = correlate(im_ref, roi_rotated, mode='same')
            correlation_max[n] = np.max(corr)
            correlation_im[n] = corr
            max_correlation_2d_pos[n, :] = np.unravel_index(np.argmax(corr, axis=None), corr.shape)

        # Plot the correlated images if the verbose option was selected
        if self.verbose:
            self.plot_correlation(correlation_im, angles)

        return correlation_max, max_correlation_2d_pos

    @log
    def optimize_rotation_translation(self, im_ref, im, downsize=False):
        """ Estimate the rotation and XY translation that align ``im`` to ``im_ref``.

        ``im_ref`` remains fixed. A central region from ``im`` is rotated over the configured angle range and cross-correlated
        with ``im_ref``. The angle and translation giving the largest correlation are stored in ``self.registration_dic``.

        Use ``apply_2d_registration`` to apply the transformation on the original image or other images.

        @param downsize: indicate whether the image was binned before being processed
        @param im_ref: reference 2D input image
        @param im: image to be aligned
        @return: the optimum angle value (returning the highest correlation value) as well as the optimum shift values
        """
        # Select the central roi of the image that needs to be realigned
        roi, x0, y0 = self.select_roi(im)
        roi_shape = roi.shape[0]

        # Calculate the multiplicative factor for the translation
        if downsize:
            downsizing_factor = 2 ** self.downsizing_power
        else:
            downsizing_factor = 1

        # Optimization is performed in two steps - the first step consists in a crude search to narrow down the optimum
        # angle for the rotation
        n_angles = (self.angles_range[1] - self.angles_range[0]) / self.crude_search_angle
        n_angles = np.ceil(n_angles).astype('uint8')
        angles_crude_search = np.linspace(self.angles_range[0], self.angles_range[1], num=int(n_angles),
                                          endpoint=False,
                                          retstep=False,
                                          dtype=None, axis=0)
        correlation_max, max_correlation_2d_pos = self.angle_search(angles_crude_search, n_angles, roi, roi_shape,
                                                                    im_ref)

        # A second search is performed between the bounds defined earlier with a step of 1°
        starting_angle = np.round(angles_crude_search[np.argmax(correlation_max)])
        angles_fine_search = np.arange(starting_angle - (self.crude_search_angle - 1),
                                       starting_angle + self.crude_search_angle, 1)
        n_angles = len(angles_fine_search)
        correlation_max, max_correlation_2d_pos = self.angle_search(angles_fine_search, n_angles, roi, roi_shape,
                                                                    im_ref)

        # Return the angle with the best correlation score as well as the shift dx, dy values - update the registration
        # dictionary
        max_correlation_idx = np.argmax(correlation_max)
        self.registration_dic["rotation"] = angles_fine_search[max_correlation_idx]
        dx = (x0 - max_correlation_2d_pos[max_correlation_idx, 0]) * downsizing_factor
        dy = (y0 - max_correlation_2d_pos[max_correlation_idx, 1]) * downsizing_factor
        self.registration_dic["shift"] = [dx, dy]

        return np.max(correlation_max)

    @staticmethod
    def im_dilate(im, zoom_factor):
        """ Perform dilation on input image according to the specified zoom factors. The output image must have
        the same size that the input image. Therefore, if zoom-factor > 1, the output image is cropped accordingly. If
        zoom-factor < 1, zero padding is applied to the output image.

        @param im: input image on which dilatation is applied
        @param zoom_factor: tuple containing the dilation factors along X and Y
        @return: output image after applying dilatation. It has the same size that the input image im.
        """
        original_shape = np.array(im.shape)
        zoomed_im = zoom(im, zoom=zoom_factor, order=0, mode="nearest")

        zoomed_shape = np.array(zoomed_im.shape)
        output_im = np.zeros_like(im)

        # Calculate coordinates for pasting/cropping centered
        min_shape = np.minimum(original_shape, zoomed_shape)
        crop_start_zoom = ((zoomed_shape - min_shape) // 2).astype(int)
        crop_start_out = ((original_shape - min_shape) // 2).astype(int)

        # Insert the valid region
        output_im[crop_start_out[0]:crop_start_out[0] + min_shape[0],
                  crop_start_out[1]:crop_start_out[1] + min_shape[1]] = zoomed_im[crop_start_zoom[0]:crop_start_zoom[0] + min_shape[0],
                                                                        crop_start_zoom[1]:crop_start_zoom[1] + min_shape[1]]

        # dilated_im = zoom(im, zoom=zoom_factor)
        # lx, ly = im.shape
        # lx_zoom, ly_zoom = dilated_im.shape
        #
        # if (lx_zoom >= lx) and (ly_zoom >= ly):
        #     dilated_im = dilated_im[int((lx_zoom - lx) / 2):int((lx_zoom - lx) / 2) + lx,
        #                             int((ly_zoom - ly) / 2):int((ly_zoom - ly) / 2) + ly]
        # else:
        #     template = np.zeros((lx, ly))
        #     template[int((lx - lx_zoom) / 2):int((lx - lx_zoom) / 2) + lx_zoom,
        #              int((ly - ly_zoom) / 2):int((ly - ly_zoom) / 2) + ly_zoom] = dilated_im
        #     dilated_im = template
        #
        # if dilated_im.shape != im.shape:
        #     print(f'lx={lx}, ly={ly}')
        #     print(f'lx_zoom={lx_zoom}, ly_zoom={ly_zoom}')

        return output_im

    # def optimize_dilatation(self, im_ref, im):
    #     """ Perform zoom transformation to optimize image registration. Since the optimum transformation is not always
    #     symmetric, the zoom factor is optimize in 2D using a grid search algorithm (step of 2%). The best parameters are
    #     saved in the registration dictionary.
    #
    #     @param im_ref: reference image to use for registration
    #     @param im: im to modify to optimize the registration
    #     @return: the best combination of zoom parameters
    #     """
    #     zoom_vals = np.linspace(self.zoom_range[0], self.zoom_range[1], num=10)
    #     correlation_max = np.zeros((len(zoom_vals), len(zoom_vals)))
    #
    #     # For each dilatation factor, perform a dilatation of the image and calculate the correlation
    #     for i, zx in enumerate(tqdm(zoom_vals)):
    #         for j, zy in enumerate(zoom_vals):
    #             dilated_im = self.im_dilate(im, (zx, zy))
    #             roi, x0, y0 = self.select_roi(dilated_im)
    #             # corr = self.optimize_rotation_translation
    #             corr, _ = self.angle_search(np.arange(-1, 2, 1), 3, roi, roi.shape[0], im_ref)
    #             correlation_max[i, j] = np.max(corr)
    #
    #     # Return the dilatation factor with the best correlation score as well as the shift dx, dy values. Update the
    #     # registration dictionary
    #     idx_max = np.unravel_index(np.argmax(correlation_max), correlation_max.shape)
    #     self.registration_dic["zoom"] = [zoom_vals[idx_max[0]], zoom_vals[idx_max[1]]]
    #
    #     return np.max(correlation_max)

    @staticmethod
    def match_image_size(im1, target_shape):
        """Crop or pad im1 to match target_shape (centered)"""
        result = np.zeros(target_shape, dtype=im1.dtype)
        x_start = max((target_shape[0] - im1.shape[0]) // 2, 0)
        y_start = max((target_shape[1] - im1.shape[1]) // 2, 0)
        x_end = x_start + min(im1.shape[0], target_shape[0])
        y_end = y_start + min(im1.shape[1], target_shape[1])

        im_x_start = max((im1.shape[0] - target_shape[0]) // 2, 0)
        im_y_start = max((im1.shape[1] - target_shape[1]) // 2, 0)
        im_x_end = im_x_start + (x_end - x_start)
        im_y_end = im_y_start + (y_end - y_start)

        result[x_start:x_end, y_start:y_end] = im1[im_x_start:im_x_end, im_y_start:im_y_end]
        return result

    @log
    def optimize_zoom(self, im_ref, im):
        """ Perform zoom transformation to optimize image registration. Since the optimum transformation is not always
        symmetric, the zoom factor is optimize in 2D. Originally, a grid search algorithm was used but was extremely
        slow. Other strategies were tested and Differential Evolution seems to provide the most robust results so far.
        The best parameters are saved in the registration dictionary.

        @param im_ref: reference image to use for registration
        @param im: im to modify to optimize the registration
        @return: the best combination of zoom parameters
        """
        def build_zoom_loss_fn(target, ref):
            """
            Returns a zoom loss function that compares target to ref images. Uses max normalized cross-correlation
            (FFT-based).
            """
            def loss_fn(z):
                try:
                    zoomed = zoom(target, zoom=z, order=1)
                    zoomed = self.match_image_size(zoomed, ref.shape)
                    corr = self.cross_correlation_fft(ref, zoomed)
                    return -np.max(corr)
                except Exception:
                    return 1e6

            return loss_fn

        # compute binned images for initial guess of the zoom factor
        resize_shape = (int(im.shape[0] / 4), int(im.shape[1] / 4))
        shape = (resize_shape[0], im.shape[0] // resize_shape[0],
                 resize_shape[1], im.shape[1] // resize_shape[1])
        im_binned = im.reshape(shape).mean(-1).mean(1)
        im_ref_binned = im_ref.reshape(shape).mean(-1).mean(1)

        # use differential evolution on the binned image
        coarse_loss_fn = build_zoom_loss_fn(im_binned, im_ref_binned)
        bounds = [self.zoom_range, self.zoom_range]
        result_de = differential_evolution(
            coarse_loss_fn,
            bounds=bounds,
            strategy='best1bin',
            popsize=15,
            tol=1e-6,
            maxiter=50
        )

        # Optimize the zoom on the full size images
        initial = result_de.x
        fine_loss_fn = build_zoom_loss_fn(im, im_ref)
        result = minimize(
            fine_loss_fn, initial, method='L-BFGS-B', bounds=bounds,
            options={'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 100},
            jac=None,
            tol=None
        )

        # Store result
        self.registration_dic["zoom"] = [float(result.x[0]), float(result.x[1])]

        return -result.fun  # higher correlation (lower error) is better

    @staticmethod
    def select_roi(im):
        """ Select the central roi of the input image.

        @param im: input 2D image
        @return: roi : the central part of the input image
        """
        lx, ly = im.shape
        x0 = int(lx / 2) - 1
        y0 = int(ly / 2) - 1
        d = int(lx / 2)
        roi = im[x0 - d // 2:x0 + d // 2, y0 - d // 2:y0 + d // 2]
        # roi = (roi - np.mean(roi)) / np.std(roi)

        return roi, x0, y0

    @log
    def optimize_3d_shift(self, stack_ref, stack, downsize):
        """ Estimate only the relative displacement along the Z axis.
        XY rotation, translation, and zoom must already have been applied. The method calculates
        XZ and YZ maximum-intensity projections from several spatial regions and estimates their Z offsets by
        cross-correlation. The mean offset is stored in image planes as ``registration_dic["z_shift"]``.

        @param stack_ref: reference 3d stack of images
        @param stack: 3d stack of images to register - a first step registration should have been applied already for
        correcting rotation, shift and zoom.
        @param downsize: indicate whether binning was applied to the input stack
        """
        width = 50
        lz, lx, ly = stack.shape
        dz = []
        for n in range(11):
            x0 = (lx - width) // 2 + (n - 5) * width
            y0 = (ly - width) // 2 + (n - 5) * width
            yz_target = np.max(stack[:, x0 - width:x0 + width, :], axis=1)
            yz_target = (yz_target - np.mean(yz_target)) / (np.std(yz_target) + 1e-8)
            xz_target = np.max(stack[:, :, y0 - width:y0 + width], axis=2)
            xz_target = (xz_target - np.mean(xz_target)) / (np.std(xz_target) + 1e-8)
            yz_ref = np.max(stack_ref[:, x0 - width:x0 + width, :], axis=1)
            yz_ref = (yz_ref - np.mean(yz_ref)) / (np.std(yz_ref) + 1e-8)
            xz_ref = np.max(stack_ref[:, :, y0 - width:y0 + width], axis=2)
            xz_ref = (xz_ref - np.mean(xz_ref)) / (np.std(xz_ref) + 1e-8)

            yz_corr = fftconvolve(yz_ref, yz_target[::-1, ::-1], mode='full')
            z_peak, _ = np.unravel_index(np.argmax(yz_corr), yz_corr.shape)
            dz.append(z_peak - (yz_target.shape[0] - 1))

            xz_corr = fftconvolve(xz_ref, xz_target[::-1, ::-1], mode='full')
            z_peak, _ = np.unravel_index(np.argmax(xz_corr), xz_corr.shape)
            dz.append(z_peak - (xz_target.shape[0] - 1))

        dz = np.mean(np.array(dz))
        dz = np.round(dz, decimals=1)
        self.registration_dic["z_shift"] = dz

        # processed_stack = np.zeros(stack.shape)
        # processed_stack_ref = np.zeros(stack_ref.shape)
        # # normalize stacks - each image of the stack is normalized separately to make sure the background is properly
        # # removed. This is an important step, in order to avoid any bias induced by the variation of fluorescence
        # # intensity within the stack. Note that the two stacks are sometimes of different shape.
        # lz, _, _ = stack.shape
        # for frame in range(lz):
        #     processed_stack[frame, :, :] = self.process_image(stack[frame, :, :], downsize=downsize)
        #
        # lz_ref, _, _ = stack_ref.shape
        # for frame in range(lz_ref):
        #     processed_stack_ref[frame, :, :] = self.process_image(stack_ref[frame, :, :], downsize=downsize)
        #
        # # # Select the central roi of the image that needs to be realigned
        # # roi_stack, _, _, z0 = self.select_roi_stack(processed_stack)
        # #
        # # # Calculate 3d correlation of the two stacks, keeping the reference stack with its original size
        # # corr = correlate(processed_stack_ref, roi_stack, mode='same')
        # # max_pos = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
        # # dz = z0 - max_pos[0]
        #
        # roi_stack, _, _, _ = self.select_roi_stack(stack)
        # roi_stack_ref, _, _, _ = self.select_roi_stack(stack_ref, dz=roi_stack.shape[0])
        # shift_zyx, error, _ = phase_cross_correlation(roi_stack_ref, roi_stack, upsample_factor=4)
        # print(f"3D zxy shift was {shift_zyx} nm.")

        # save dz
        # self.registration_dic["z_shift"] = float(shift_zyx[0])

    @staticmethod
    def select_roi_stack(stack, dz=None):
        """ Select the central roi of the input image.

        @param dz: (int) when indicated, fix the number of planes for the roi
        @param stack: input 3d stack of images
        @return: roi : the central part of the input stack
        """
        lz, lx, ly = stack.shape
        z0 = int(lz / 2) - 1
        x0 = int(lx / 2) - 1
        y0 = int(ly / 2) - 1
        dxy = int(lx / 2)
        if dz is None:
            dz = int(lz / 2)
        else:
            dz = dz

        roi = stack[z0 - dz // 2:z0 + dz // 2,
                    x0 - dxy // 2:x0 + dxy // 2,
                    y0 - dxy // 2:y0 + dxy // 2]
        #roi = (roi - np.mean(roi)) / np.std(roi)

        return roi, x0, y0, z0

    def apply_2d_registration(self, im, order=1, downsize=False):
        """Apply the stored 2D transformation to an image.
        Operations are applied to the target image in this order:
            1. rotation;
            2. translation;
            3. independent zoom along axes 0 and 1;
            4. central cropping or padding to restore the original image shape.

        @param order: (int) indicate the order of the spline interpolation
        @param im: 2D image (np array)
        @param downsize: indicate whether the input image was binned
        @return: im_registered, the modified image.
        """

        # Registration dictionary
        optimum_angle = self.registration_dic["rotation"]
        dx, dy = self.registration_dic["shift"]
        optimum_zoom = self.registration_dic["zoom"]

        # If the selected image was binned, adapt the shift corrections
        if downsize:
            downsizing_factor = 2 ** self.downsizing_power
            dx = dx / downsizing_factor
            dy = dy / downsizing_factor

        # Perform registration according to the parameters
        im_rotated = rotate(im, optimum_angle, axes=(1, 0), reshape=False, order=order, mode='constant', cval=0.0)
        im_rotated_shifted = shift(im_rotated, [-dx, -dy], order=order, mode='constant')

        im_zoomed = zoom(im_rotated_shifted, zoom=optimum_zoom, order=order, mode='constant')
        im_registered = self.match_image_size(im_zoomed, im.shape)

        return im_registered

    def apply_3d_registration(self, stack, order=1, downsize=False):
        """ Recalculate the input stack after applying registration

        @param stack: stack to which registration will be applied
        @param order: (int) indicate the order of the spline interpolation
        @param downsize: indicate whether binning was applied to the input stack
        @return: return the registered stack
        """
        lz, lx, ly = stack.shape
        aligned_stack_2d = np.zeros((lz, lx, ly))

        # for each image, apply the 2d registration optimized for the dataset
        for frame in tqdm(range(lz)):
            aligned_stack_2d[frame, :, :] = self.apply_2d_registration(stack[frame, :, :],
                                                                       order=order,
                                                                       downsize=downsize)

        # apply the z shift
        dz = self.registration_dic["z_shift"]
        aligned_stack_3d = shift(aligned_stack_2d, [dz, 0, 0], order=order, mode='constant')

        return aligned_stack_3d

    def save_tiff_image(self, im, name, rescale=False):
        """ Save input image as a tiff file.

        @param rescale: boolean - indicate whether the intensity should be rescaled before saving the tif
        @param im: input 2D images
        @param name: saving name
        """
        im_path = os.path.join(self.saving_path, name)
        if rescale:
            im_min = np.min(im)
            im_max = np.max(im)
            im = (2 ** 16 - 1) * (im - im_min) / (im_max - im_min)

        with TiffWriter(im_path) as tif:
            tif.write(im.astype(np.uint16))

    def save_registration_dic(self, prefix=''):
        """ Save in a json file the dictionary containing the optimized parameters for registration
        """
        registration_file = os.path.join(self.saving_path, f'{prefix}_registration_parameters.json')
        with open(registration_file, 'w') as file:
            json.dump(self.registration_dic, file)

    def load_registration_dic(self, folder):
        """ Load the json file containing the dictionary with the optimized parameters for registration
        @param folder: folder where to look for file where the registration parameters were saved
        """
        f = open(folder)
        dic = json.load(f)

        required_keys = {"shift", "rotation", "zoom", "z_shift"}
        if not isinstance(dic, dict):
            raise ValueError("Loaded registration file is not a dictionary.")

        missing = required_keys - dic.keys()
        if missing:
            raise ValueError(f"Registration file is missing keys: {missing}")

        self.registration_dic = dic

    @staticmethod
    def plot_correlation(im, angles):
        """ Plot all the correlated images if the verbose option is selected.

        @param im: array of images
        @param angles: array of angles tested for the correlation
        """
        plt.figure(figsize=(16, 10))
        num_im = im.shape[0]

        subplot_x = np.round(np.sqrt(num_im))
        subplot_y = np.ceil(num_im / subplot_x)

        for n in range(num_im):
            plt.subplot(subplot_x, subplot_y, n + 1)
            plt.imshow(im[n], cmap='gray')
            plt.title(f'{angles[n]}°')
            plt.axis('off')

        plt.show()

    def rescale_contrast(self, im):
        """ This function is used to improve the contrast of an input image for the png images. The input image is first
        processed and the intensity is rescaled in a second step.

        @param im: input 2D image (np array)
        @return: 2D image with rescaled intensity
        """
        im = np.divide(im, gaussian_filter(im, self.gaussian_filter_size))
        intensity = np.copy(im)
        intensity = np.reshape(intensity, (intensity.shape[0] * intensity.shape[1], 1))
        intensity = np.sort(intensity, axis=0)
        int_min = np.percentile(intensity, 0.5)
        int_max = np.percentile(intensity, 99.5)
        im = (2 ** 16 - 1) * (im - int_min) / (int_max - int_min)
        im[im < 0] = 0
        im[im > 2 ** 16 - 1] = 2 ** 16
        return im

    def save_aligned_overlay(self, im_ref, im_aligned, im_name="MIP"):
        """ Overlay the registered image with the reference and save the result

        @param im_ref: reference 2D image
        @param im_aligned: aligned image
        @param im_name: name of the image to save
        """

        # Registration dictionary
        optimum_angle = np.around(self.registration_dic["rotation"], decimals=1)
        dx, dy = self.registration_dic["shift"]
        optimum_zoom = self.registration_dic["zoom"]

        # Rescale contrast for the original images
        # im_ref_rescaled = self.rescale_contrast(im_ref)

        overlay = np.zeros((im_ref.shape[0], im_aligned.shape[1], 3))  # Create an empty RGB image
        overlay[..., 0] = im_ref  # Map image1 to the red channel
        overlay[..., 1] = im_aligned  # Map image2 to the green channel

        # Save the aligned montage :
        im_title = im_name + '_registered_montage.png'
        im_path = os.path.join(self.saving_path, im_title)
        plt.figure(figsize=(16, 10))
        plt.imshow(overlay/255)
        plt.title(f'Aligned images - angle {optimum_angle}° \nshift {dx, dy}px - '
                  f'zoom {np.round(optimum_zoom, decimals=3)}')
        plt.axis('off')
        plt.savefig(im_path, dpi=250)

    def save_aligned_overlay_xz(self, stack_ref, stack_aligned, im_name="MIP"):
        """ Overlay the registered image with the reference and save the result

        @param stack_ref: (numpy array) reference 3D stack
        @param stack_aligned: (numpy array) aligned 3D stack
        @param im_name: name of the image to save
        """

        # Registration dictionary
        optimum_angle = np.around(self.registration_dic["rotation"], decimals=1)
        dx, dy = self.registration_dic["shift"]
        optimum_zoom = self.registration_dic["zoom"]
        z_shift = self.registration_dic["z_shift"]

        # Plot sections of XZ planes
        width = 50
        lz, lx, ly = stack_aligned.shape
        lz_ref, _, _ = stack_ref.shape
        lz = np.min((lz, lz_ref))

        fig, axs = plt.subplots(nrows=5, ncols=1)
        for n, ax in enumerate(axs):
            x0 = (lx - width) // 2 + (n - 2) * width
            mip_aligned = np.max(stack_aligned[:lz, x0 - width:x0 + width, :], axis=1)
            mip_ref = np.max(stack_ref[:lz, x0 - width:x0 + width, :], axis=1)

            mip_aligned = self.convert_to_8bit(mip_aligned)
            mip_ref = self.convert_to_8bit(mip_ref)

            overlay = np.zeros((mip_ref.shape[0], mip_ref.shape[1], 3))  # Create an empty RGB image
            overlay[..., 0] = mip_ref  # Map image1 to the red channel
            overlay[..., 1] = mip_aligned  # Map image2 to the green channel

            ref_rgb = np.stack([mip_ref] * 3, axis=-1)
            aligned_rgb = np.stack([mip_aligned] * 3, axis=-1)
            composite = np.vstack([ref_rgb, overlay, aligned_rgb])
            ax.imshow(composite / 255)
            ax.axis('off')

        # Save the aligned montage :
        im_title = im_name + '_registered_montages_XZ.png'
        im_path = os.path.join(self.saving_path, im_title)
        plt.suptitle(f'Aligned images - angle {optimum_angle}° \nshift {dx, dy}px - '
                  f'zoom {np.round(optimum_zoom, decimals=3)} - z_shift {z_shift} frames')
        plt.savefig(im_path, dpi=250)

    def save_aligned_montage(self, im_ref, im_2_align, im_aligned, im_name="MIP"):
        """ Save the results

        @param im_ref: reference 2D image
        @param im_2_align: image to align
        @param im_aligned: aligned image
        @param im_name: name of the image to save
        """
        fig = plt.figure(figsize=(16, 10))

        # Registration dictionary
        optimum_angle = np.around(self.registration_dic["rotation"], decimals=1)
        dx, dy = self.registration_dic["shift"]
        optimum_zoom = self.registration_dic["zoom"]

        # Save the images as a subplot
        plt.subplot(131)
        plt.imshow(im_ref, cmap='gray')
        plt.title('Reference')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(im_2_align, cmap='gray')
        plt.title('Original image to align')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(im_ref, cmap='Blues')
        plt.imshow(im_aligned, cmap='Reds', alpha=0.5)
        plt.title(f'Aligned images - angle {optimum_angle}° \nshift {dx, dy}px - '
                  f'zoom {np.round(optimum_zoom, decimals=3)}')
        plt.axis('off')

        im_title = im_name + '_registered.png'
        im_path = os.path.join(self.saving_path, im_title)
        plt.savefig(im_path, dpi=150)
        plt.close(fig)

    def simulate_data(self, stack, shift_range, angle_range, zoom_range, shift_3d_range, order, output_name):

        # from the original stack, remove the 5 first and last frames (for the simulations of the z-shift)
        lz = stack.shape[0]
        reduced_stack = stack[shift_3d_range: lz - shift_3d_range, :, :]

        # from the parameters, randomly create a set of transformations
        im_zoom = np.random.uniform(low=zoom_range[0], high=zoom_range[1], size=(2,))
        # im_zoom = np.random.uniform(low=zoom_range[0], high=zoom_range[1])
        im_shift = np.random.uniform(low=-shift_range, high=shift_range, size=(2,))
        im_rotation = np.random.uniform(low=-angle_range, high=angle_range)
        im_shift_3d = np.random.uniform(low=-shift_3d_range, high=shift_3d_range)

        # save the parameters in the registration dictionary.
        self.registration_dic: dict = {"shift": list(np.round(im_shift)),
                                       "rotation": np.round(im_rotation),
                                       # "zoom": [np.around(im_zoom, decimals=2), np.around(im_zoom, decimals=2)],
                                       "zoom": list(np.around(im_zoom, decimals=2)),
                                       "z_shift": np.round(im_shift_3d)}
        print(self.registration_dic)

        # Save the registration dictionary as a json file
        self.save_registration_dic(prefix='simulated_')

        # apply the simulated parameters to the input stack and save the result. The z-shift is set to zero. It will be
        # applied manually later in order to avoid the creation of empty frames
        self.registration_dic["z_shift"] = 0
        simulated_stack = self.apply_3d_registration(stack, order=order, downsize=False)

        # apply the z-shift to the simulated stack
        simulated_stack = simulated_stack[int(shift_3d_range + np.round(im_shift_3d)):
                                          int(lz - shift_3d_range + np.round(im_shift_3d)), :, :]

        # # replace all pixel equal to zero by a random noise typical for the sCMOS
        # lz, lx, ly = simulated_stack.shape
        # simulated_stack = np.reshape(simulated_stack, (lz * lx * ly, ))
        # idx = np.argwhere(simulated_stack == 0)
        # noise = np.random.normal(loc=10, scale=1, size=(idx.shape[0],)) + \
        #         np.random.normal(loc=100, scale=1.2, size=(idx.shape[0],))
        # simulated_stack[idx[:, 0]] = noise
        # simulated_stack = np.reshape(simulated_stack, (lz, lx, ly))

        # replace all pixels equal to zero by 1
        simulated_stack[simulated_stack == 0] = 1

        # reset the registration dictionary
        self.registration_dic: dict = {"shift": [0, 0],
                                       "rotation": 0,
                                       "zoom": [1, 1],
                                       "z_shift": 0}

        # save the simulated stack
        # self.save_tiff_image(simulated_stack, output_name, rescale=False)
        return reduced_stack, simulated_stack

    def run_registration(self, order=1, downsize=False, full_3D=False):
        """     Execute the registration workflow defined by the YAML configuration.

        The function locates the reference and target TIFF stacks, extracts the configured DAPI channels,
        estimates the target-to-reference transform, generates registration diagnostics, and writes the fitted parameters to JSON.

        @param order: (int) indicate the order for the spline interpolation
        @param downsize: (bool) indicate whether the analysis will be performed on the raw image or a binned image
        @param full_3D: (bool) indicate whether registration should be computed in 3D or only in 2D
        """
        # define the path to the images
        path_ref = glob(os.path.join(self.path_image_ref, self.template_ref_file))
        path_to_align = glob(os.path.join(self.path_image_to_align, self.template_target_file))

        # Load the reference image
        stack_reference = self.load_stack(path_ref, self.ref_total_number_channels, self.ref_channel)
        stack_to_align = self.load_stack(path_to_align, self.align_total_number_channels, self.align_channel)

        # Calculate the MIP for the two images
        mip_reference = self.mip(stack_reference)
        mip_to_align = self.mip(stack_to_align)

        # Compute the normalized and 8-bit images
        mip_reference_8bit = self.convert_to_8bit(mip_reference)
        mip_to_align_8bit = self.convert_to_8bit(mip_to_align)

        # Process the two images
        if downsize:
            mip_to_align = self.bin_image(mip_to_align)
            mip_reference = self.bin_image(mip_reference)

        mip_to_align_processed = self.process_image(mip_to_align, downsize=downsize)
        mip_reference_processed = self.process_image(mip_reference, downsize=downsize)

        # Perform a first optimization, correcting rotation & shift
        self.optimize_rotation_translation(mip_reference_processed, mip_to_align_processed, downsize=downsize)

        # If the option was selected, perform a second optimization using a zoom morphological transformation. This
        # optimization is performed in 2 steps :
        # 1- the optimized shift and rotation are first applied
        # 2- the dilatation is then optimized by performing a grid search
        # 3- knowing the optimum dilatation, a new optimization of the rotation & translation is performed
        if self.apply_zoom:
            mip_registered = self.apply_2d_registration(mip_to_align_processed, order=order, downsize=downsize)
            self.optimize_zoom(mip_reference_processed, mip_registered)
            mip_registered = self.im_dilate(mip_to_align_processed, self.registration_dic['zoom'])
            self.optimize_rotation_translation(mip_reference_processed, mip_registered, downsize=downsize)

        # Apply the transformation to the original image - so downsize is set to False
        im_registered_final = self.apply_2d_registration(mip_to_align_8bit, order=order)

        # Plot and save the results for registration based only on rotation and shift morphological transformations
        self.save_aligned_montage(mip_reference_8bit, mip_to_align_8bit, im_registered_final,
                                  im_name=f"MIP_registered_{self.data_tag}")
        self.save_aligned_overlay(mip_reference_8bit, im_registered_final, im_name=f"MIP_registered_{self.data_tag}")

        # Register the full stack & perform a final alignment along the z-axis. Plot the XZ and ZY registered stack
        if full_3D:
            if downsize:
                binned_stack_to_align = self.bin_stack(stack_to_align)
                binned_stack_reference = self.bin_stack(stack_reference)
                stack_registered = self.apply_3d_registration(binned_stack_to_align, order=order, downsize=downsize)
                self.optimize_3d_shift(binned_stack_reference, stack_registered, downsize=downsize)
            else:
                stack_registered = self.apply_3d_registration(stack_to_align, order=order, downsize=downsize)
                self.optimize_3d_shift(stack_reference, stack_registered, downsize=downsize)

        # Save the registration dic
        self.save_registration_dic(prefix=self.data_tag)
        print(self.registration_dic)

        # Final registration of the stack and saving
        stack_registered_final = self.apply_3d_registration(stack_to_align, order=order, downsize=False)
        # saving_filename = f"{Path(path_to_align[0]).stem}_registred.tif"
        # self.save_tiff_image(stack_registered_final, saving_filename, rescale=False)

        # Plot the XZ and ZY registered stack
        if full_3D:
            self.save_aligned_overlay_xz(stack_reference, stack_registered_final,
                                         im_name=f"MIP_registered_{self.data_tag}")


if __name__ == "__main__":

    args = ImageRegistration.parse_arguments()

    align = ImageRegistration(args.config_file)

    align.run_registration(
        order=1,
        downsize=True,
        full_3D=False,
    )

    # for n_simu in range(50):
    #
    #     # Instantiate the alignment class
    #     _align = ImageRegistration(downsizing_power=1, gaussian_filter_size=10, angles_range=(-21, 21),
    #                                zoom_range=(1, 1.15), apply_zoom=True, verbose=False)
    #
    #     # Define and create the saving folder
    #     simulation_folder = f'{str(n_simu).zfill(3)}_simulation'
    #     _align.saving_path = os.path.join(dest_folder, simulation_folder)
    #     if os.path.isdir(_align.saving_path):
    #         shutil.rmtree(_align.saving_path)
    #     os.mkdir(_align.saving_path)
    #
    #     # Load the reference image
    #     stack_reference = _align.load_stack(path_image_ref, ref_total_number_channels, ref_channel)
    #     # stack_to_align = _align.load_stack(path_image_to_align, align_total_number_channels, align_channel)
    #
    #     # Create the simulated image by randomly defining the transformation parameters
    #     stack_reference, stack_to_align = _align.simulate_data(stack_reference, 50, 20, [0.9, 1], 5, order,
    #     'simulation.tif')
    #
    #     # Calculate the MIP for the two images
    #     mip_reference = _align.mip(stack_reference)
    #     mip_to_align = _align.mip(stack_to_align)
    #
    #     Lx, Ly = mip_reference.shape
    #     # _align.save_tiff_image(mip_reference, "MIP_ref")
    #     # _align.save_tiff_image(mip_to_align, "MIP_to_align")
    #
    #     # Process the two images
    #     mip_to_align_binned = _align.bin_image(mip_to_align)
    #     mip_to_align_processed = _align.process_image(mip_to_align_binned, downsize=True)
    #     mip_reference_binned = _align.bin_image(mip_reference)
    #     mip_reference_processed = _align.process_image(mip_reference_binned, downsize=True)
    #
    #     # # mip_to_align_binned = _align.bin_image(mip_to_align)
    #     # mip_to_align_processed = _align.process_image(mip_to_align, downsize=False)
    #     # # mip_reference_binned = _align.bin_image(mip_reference)
    #     # mip_reference_processed = _align.process_image(mip_reference, downsize=False)
    #
    #     # Perform a first optimization, correcting rotation & shift
    #     _align.optimize_rotation_translation(mip_reference_processed, mip_to_align_processed, downsize=True)
    #
    #     # If the option was selected, perform a second optimization using a zoom morphological transformation. This
    #     # optimization is performed in 2 steps :
    #     # 1- the optimized shift and rotation are first applied
    #     # 2- the dilatation is then optimized by performing a grid search
    #     # 3- knowing the optimum dilatation, a new optimization of the rotation & translation is performed
    #     if _align.apply_zoom:
    #         mip_registered = _align.apply_2d_registration(mip_to_align_processed, downsize=True)
    #         _align.optimize_dilatation(mip_reference_processed, mip_registered)
    #         mip_registered = _align.im_dilate(mip_to_align_processed, _align.registration_dic['zoom'])
    #         _align.optimize_rotation_translation(mip_reference_processed, mip_registered, downsize=True)
    #
    #     # Apply the transformation to the original image
    #     im_registered_final = _align.apply_2d_registration(_align.process_image(mip_to_align))
    #
    #     # Plot and save the results for registration based only on rotation and shift morphological transformations
    #     _align.save_aligned_montage(mip_reference, mip_to_align, im_registered_final, im_name="MIP_registered")
    #     _align.save_aligned_overlay(mip_reference, im_registered_final, im_name="MIP_registered")
    #
    #     # Register the full stack
    #     binned_stack_to_align = _align.bin_stack(stack_to_align)
    #     binned_stack_reference = _align.bin_stack(stack_reference)
    #     stack_registered = _align.apply_3d_registration(binned_stack_to_align, downsize=True)
    #
    #     # Perform a final alignment along the z-axis
    #     _align.optimize_3d_shift(binned_stack_reference, stack_registered, downsize=True)
    #
    #     # Final registration of the stack and saving
    #     stack_registered_final = _align.apply_3d_registration(stack_to_align, downsize=False)
    #     _align.save_tiff_image(stack_registered_final, "final_registered_stack", rescale=False)
    #
    #     # Calculate the correlation, taking into account only the central part of the image
    #     print('Calculate final correlations ...')
    #
    #     nframes = stack_reference.shape[0]
    #     correlation = np.zeros((nframes, 2))
    #     for nframe in tqdm(range(nframes)):
    #         stack_reference_roi = stack_reference[nframe, 512: 1536, 512: 1536]
    #         stack_reference_roi = stack_reference_roi.reshape((1024 * 1024,))
    #         stack_registered_roi = stack_registered_final[nframe, 512: 1536, 512: 1536]
    #         stack_registered_roi = stack_registered_roi.reshape((1024 * 1024,))
    #         correlation[nframe, :] = pearsonr(stack_reference_roi, stack_registered_roi)
    #
    #     np.save(os.path.join(_align.saving_path, 'correlation.npy'), correlation)
    #
    #     # Save the final parameters :
    #     print(_align.registration_dic)
    #     _align.save_registration_dic()