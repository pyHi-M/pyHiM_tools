#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script filters a mask file based on user-provided morphological properties and intensity values from the original image.

The script loads an input file with masks and an optional original intensity image, then filters the masks based on:
- Minimum diameter
- Maximum diameter
- Minimum z position
- Maximum z position
- Eccentricity
- Maximum intensity in the original image

Usage:
    python filter_masks.py --input mask_file --output output_file --min_diameter 5 --max_diameter 50 --min_z 10 --max_z 100 --eccentricity 0.8 --min_intensity 1000 --original original_image_file

Example:
    python filter_masks.py --input mask.tif --output mask_filtered.tif --min_diameter 5 --max_diameter 50 --min_z 10 --max_z 100 --eccentricity 0.8 --min_intensity 1000 --original original.tif

Example with multiple input files, assuming you provide the intensity filenames, and that the masks are located in the mask_3d/data/ folder for each intensity file:
    ls *mask0*ch01* | mask_filter.py --pipe --pyHiM --min_intensity 2000 --min_z 5 --max_z 20  --replace_mask_file

Arguments:
    --input              Name of input mask file (TIFF, NPY, or HDF5 format).
    --output             Name of output file (same format as input).
    --replace_mask_file  Overwrite the input mask file with the filtered masks.
    --min_diameter       Minimum diameter of masks to keep.
    --max_diameter       Maximum diameter of masks to keep.
    --min_z              Minimum z position of masks to keep.
    --max_z              Maximum z position of masks to keep.
    --num_pixels         Minimum number of pixels to keep.
    --min_intensity      Maximum intensity of masks to keep in the original image.
    --intensity_image    Name of original intensity image file (TIFF, NPY, or HDF5 format) if min_intensity is specified.

Installation:
    conda create -y -n mask_analysis python==3.11
    pip install numpy scipy h5py tifffile scikit-image matplotlib

Created on Tue Jul  4 13:26:12 2023
Updated by OpenAI's GPT-4

Author: marcnol
"""
import os
from tifffile import imread, imwrite
import numpy as np
from skimage.measure import regionprops, label
import argparse
import h5py
import sys
import shutil
from tqdm import trange, tqdm
import select
import json
from scipy.ndimage import shift as shift_image


def read_image(file_path):
    """Read an image from a file (TIFF, NPY, or HDF5)."""
    if file_path.endswith(".h5"):
        with h5py.File(file_path, "r") as f:
            image = f["image"][:]
    elif file_path.endswith(".npy"):
        image = np.load(file_path)
    else:
        image = imread(file_path)
    return image


def save_image(image, file_path):
    """Save an image to a file (TIFF, NPY, or HDF5)."""
    if file_path.endswith(".h5"):
        with h5py.File(file_path, "w") as f:
            f.create_dataset("image", data=image, compression="gzip")
    elif file_path.endswith(".npy"):
        np.save(file_path, image)
    else:
        imwrite(file_path, image)


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", help="Name of input mask file (TIFF, NPY, or HDF5 format)."
    )
    parser.add_argument(
        "--output", help="Name of output file (same format as input).", default=None
    )
    parser.add_argument(
        "--replace_mask_file",
        action="store_true",
        help="Overwrite the input mask file with the filtered masks.",
    )
    parser.add_argument(
        "--min_diameter", type=float, help="Minimum diameter of masks to keep."
    )
    parser.add_argument(
        "--max_diameter", type=float, help="Maximum diameter of masks to keep."
    )
    parser.add_argument(
        "--min_z", type=int, help="Minimum z position of masks to keep."
    )
    parser.add_argument(
        "--max_z", type=int, help="Maximum z position of masks to keep."
    )
    parser.add_argument(
        "--num_pixels", type=float, help="Minimum number of pixels in masks to keep."
    )
    parser.add_argument(
        "--min_intensity",
        type=float,
        help="Minimum intensity of masks to keep in the original image.",
    )
    parser.add_argument(
        "--intensity_image",
        help="Name of original intensity image file (TIFF, NPY, or HDF5 format) if min_intensity is specified.",
    )
    parser.add_argument(
        "--pipe", help="inputs file list from stdin (pipe)", action="store_true"
    )
    parser.add_argument(
        "--pyHiM",
        help="Will shift and interpolate intensity images",
        action="store_true",
    )
    parser.add_argument(
        "--zBinning", default=2, type=int, help="zBinning used for mask segmentation"
    )

    args = parser.parse_args()

    if args.replace_mask_file and args.output:
        parser.error(
            "Cannot use --replace_mask_file and --output together. Choose one."
        )

    if args.pipe:
        pipe_status = True
        if select.select(
            [
                sys.stdin,
            ],
            [],
            [],
            0.0,
        )[0]:
            files = [line.rstrip("\n") for line in sys.stdin]
        else:
            parser.error("Nothing in stdin! ")
    else:
        pipe_status = False
        if not args.input:
            parser.error(
                "mask_filter.py: error: the following arguments are required: --input. Otherwise provide images using --pipe! "
            )

        files = [args.input]

    return args, files, pipe_status


def filter_masks(
    labeled_image,
    original_im=None,
    min_diameter=None,
    max_diameter=None,
    min_z=None,
    max_z=None,
    num_pixels=None,
    min_intensity=None,
):
    # removes bottom and top planes from labeled image

    print("$ filtering out bottom and top planes")
    for plane in trange(labeled_image.shape[0]):
        if (min_z is not None and plane < min_z) or (
            max_z is not None and plane > max_z
        ):
            labeled_image[plane, :, :] = 0

    print("$ filtering out min/max diameters, num_pixels, min_intensity")
    mask = np.zeros(labeled_image.shape, dtype=bool)
    regions = regionprops(
        labeled_image, intensity_image=original_im if original_im is not None else None
    )

    print(f"$ Will analyze {len(regions)} labels")
    removed = 0
    for region in tqdm(regions):
        if (
            (min_diameter is not None and region.equivalent_diameter < min_diameter)
            or (max_diameter is not None and region.equivalent_diameter > max_diameter)
            or (num_pixels is not None and region.area < num_pixels)
            or (min_intensity is not None and region.max_intensity < min_intensity)
        ):

            mask[labeled_image == region.label] = True
            removed += 1

    # Set the identified regions to zero
    labeled_image[mask] = 0
    print(f"$ Removed {removed} labels")

    return labeled_image


def _remove_z_planes(image_3d, z_range):
    """
    Removes planes in input image.
    For instance, if you provide a z_range = range(0,image_3d.shape[0],2)

    then the routine will remove any other plane. Number of planes skipped
    can be programmed by tuning z_range.

    Parameters
    ----------
    image_3d : numpy array
        input 3D image.
    z_range : range
        range of planes for the output image.

    Returns
    -------
    output: numpy array
    """
    output = np.zeros(
        (len(z_range), image_3d.shape[1], image_3d.shape[2]), dtype=image_3d.dtype
    )
    for i, index in enumerate(z_range):
        output[i, :, :] = image_3d[index, :, :]

    return output


def _shift_xy_mask_3d(image, shift):
    number_planes = image.shape[0]
    print(f"> Shifting {number_planes} planes")
    shift_3d = np.zeros((3))
    shift_3d[0], shift_3d[1], shift_3d[2] = 0, shift[0], shift[1]
    return shift_image(image, shift_3d)


def find_roi_name_in_path(file):
    return os.path.basename(file).split("_")[3]


def find_label_in_path(file):
    return os.path.basename(file).split("_")[2]


def load_json(file_name):
    """Load a JSON file like a python dict

    Parameters
    ----------
    file_name : str
        JSON file name

    Returns
    -------
    dict
        Python dict
    """
    if os.path.exists(file_name):
        with open(file_name, encoding="utf-8") as json_file:
            return json.load(json_file)
    print(f"[WARNING] path {file_name} doesn't exist!")
    raise ValueError


def load_params(folder):
    # loads dicShifts with shifts for all rois and all labels
    print(f"$ will look for the parameters file at: {folder+'parameters.json'}")
    if os.path.exists(folder + "parameters.json"):
        print("$ Loading parameters.json")
        return load_json(folder + "parameters.json")
    elif os.path.exists(folder + "infoList.json"):
        print(
            "[WARNING] 'infoList.json' is a DEPRECATED file name, please rename it 'parameters.json'"
        )
        print("$ Loading infoList.json")
        return load_json(folder + "infoList.json")
    else:
        raise ValueError("[ERROR] 'parameters.json' file not found.")


def get_dict_shifts(folder):
    params = load_params(folder)
    dict_shifts_path = (
        params["common"]["alignImages"].get("register_global_folder", "register_global")
        + os.sep
        + "data"
        + os.sep
        + params["common"]["alignImages"].get("outputFile", "shifts")
        + ".json"
    )
    print(f"$ Loading dictionary from: {folder+dict_shifts_path}")
    return load_json(folder + dict_shifts_path)


def main():
    args, files, pipe_status = parseArguments()

    print(
        "\n$ <{}> input file(s) to process=\n {}".format(
            len(files), "\n".join(map(str, files))
        )
    )

    for file in files:

        if pipe_status:
            if args.pyHiM:
                file_image = file
                file = (
                    file.replace(".npy", "_3Dmasks.npy")
                    .replace(".tif", "_3Dmasks.npy")
                    .replace(".h5", "_3Dmasks.npy")
                )
                file = os.path.join(
                    os.path.dirname(file), "mask_3d/data/", os.path.basename(file)
                )
            else:
                file_image = []

        if os.path.exists(file):
            print("=" * 80 + f"$ mask file: {file}")
        else:
            print(
                f"! Error: mask file <{file}> not found, will move on to next file in line..."
            )
            continue

        # backups file and sets saving paths
        if args.replace_mask_file:
            backup_file = file
            backup_file = (
                backup_file.replace(".tif", "_original.tif")
                .replace(".npy", "_original.npy")
                .replace(".h5", "_original.h5")
            )
            print(f"$ Creating a backup of the original mask file: {backup_file}")
            shutil.copyfile(file, backup_file)
            save_path = file
        else:
            if args.output is not None:
                save_path = (
                    file.replace(".tif", "_" + args.output + ".tif")
                    .replace(".npy", "_" + args.output + ".npy")
                    .replace(".h5", "_" + args.output + ".h5")
                )
            else:
                save_path = (
                    file.replace(".tif", "_filtered.tif")
                    .replace(".npy", "_filtered.npy")
                    .replace(".h5", "_filtered.h5")
                )

        # loads mask file
        print(f"$ Loading mask file: {file}")
        mask = read_image(file)

        # load intensity image if necessary
        original_im = None
        if args.min_intensity is not None:
            if not pipe_status:
                if not args.intensity_image:
                    print(
                        "Error: Original image file must be provided if min_intensity is specified."
                    )
                    sys.exit(1)
                else:
                    file_image = args.intensity_image

            print(f"$ Loading original intensity image file: {file_image}")
            original_im = read_image(file_image)

            if args.pyHiM:

                # shifts image
                roi_name = find_roi_name_in_path(file_image)
                label = find_label_in_path(file_image)
                # folder = os.path.dirname(file_image) + os.sep
                folder = os.path.dirname(file_image)
                folder = os.path.join(folder, "")
                dict_shifts = get_dict_shifts(folder)

                ## uses existing shift calculated by align_images
                try:
                    shift = dict_shifts[f"ROI:{roi_name}"][label]
                    print("> Applying existing XY shifts...")
                except KeyError as e:
                    shift = None
                    raise SystemExit(
                        f"# Could not find dictionary with alignment parameters for this ROI: ROI:{roi_name}, label: {label}"
                    ) from e

                original_im = _shift_xy_mask_3d(original_im, shift)

            # interpolates image
            z_range = range(0, original_im.shape[0], args.zBinning)
            original_im = _remove_z_planes(original_im, z_range)

        # filters mask file
        print("$ Filtering masks...")
        filtered_image = filter_masks(
            mask,
            original_im,
            args.min_diameter,
            args.max_diameter,
            args.min_z,
            args.max_z,
            args.num_pixels,
            args.min_intensity,
        )

        # saved filtered mask file
        print(f"$ Saving filtered masks to: {save_path}")
        save_image(filtered_image.astype(np.uint8), save_path)

    print("Finished execution")


if __name__ == "__main__":
    main()
