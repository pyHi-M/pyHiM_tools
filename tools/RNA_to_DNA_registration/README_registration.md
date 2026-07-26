# DAPI image registration

## Overview

`Image_registration.py` registers a **target DAPI image stack** onto a **reference DAPI image stack**.

The script was designed for experiments in which the same field of view is imaged at two different time points, for example before and after disassembly and reassembly of a fluidics chamber. A typical application is the alignment of DNA-HiM and RNA-imaging experiments so that DNA traces, segmented cells, and RNA measurements can be compared in a common coordinate system.

The reference image remains fixed. The target image is transformed to match it.

The estimated transformation can include:

- in-plane rotation;
- translation along the two image axes;
- optional independent zoom along the two image axes;
- optional displacement along the Z axis.

The current implementation performs a **global registration**. It does not correct local or non-rigid deformation.

---

## Registration workflow

The script performs the following steps:

1. Finds one reference TIFF file and one target TIFF file from the paths and filename patterns in the YAML configuration.
2. Extracts the selected DAPI channel from each interleaved multichannel stack.
3. Calculates a maximum-intensity projection along Z.
4. Corrects gradual illumination differences and standardizes image intensities.
5. Estimates the rotation and XY translation that maximize image cross-correlation.
6. Optionally estimates independent zoom factors along the two image axes.
7. Optionally estimates a global Z displacement.
8. Saves the registration parameters and diagnostic images.

The transformation direction is:

```text
target-image coordinates -> reference-image coordinates
```

---

## Requirements

The script requires Python 3 and the following packages:

```bash
pip install numpy scipy scikit-image tifffile tqdm matplotlib pyyaml
```

The script uses Matplotlib's non-interactive `Agg` backend, so diagnostic figures can be saved when running on a computer without a graphical display.

---

## Input-image requirements

### Interleaved multichannel stacks

The TIFF stacks are assumed to contain interleaved channels in the following order:

```text
Z0/channel 0
Z0/channel 1
...
Z1/channel 0
Z1/channel 1
...
```

Channel indices use Python's zero-based convention:

```text
0 = first channel
1 = second channel
2 = third channel
```

For example, for a two-channel stack in which DAPI is the second channel:

```yaml
number_channel_reference_stack: 2
reference_channel: 1
```

---

## Configuration file

The script is configured through `Config_registration.yml`.

A minimal example is shown below:

```yaml
data_tag: "ROI008_DNA_RNA"

registration_parameters:
  downsizing_power: 1
  gaussian_filter_size: 10
  angles_range: [-20, 20]
  zoom_range: [1.0, 1.1]
  apply_zoom: false
  verbose: false

reference_DAPI_folder: "/path/to/reference/folder"
template_reference_DAPI_file: "*008_ROI.tif"
number_channel_reference_stack: 2
reference_channel: 1

target_DAPI_folder: "/path/to/target/folder"
template_target_DAPI_file: "*008_ROI.tif"
number_channel_target_stack: 2
target_channel: 1

dest_root_folder: "/path/to/output/root"
dest_analysis_folder: "registration_results"
overwrite_existing_file: true
```

### Parameter reference

| Parameter | Type | Meaning |
|---|---|---|
| `data_tag` | string | Short identifier added to output filenames. Prefer letters, numbers, underscores, and hyphens. |
| `registration_parameters.downsizing_power` | integer | Images used for parameter estimation are reduced by a factor of `2**downsizing_power` along each image axis. |
| `registration_parameters.gaussian_filter_size` | number | Width of the Gaussian filter used to estimate and remove gradual illumination variation. |
| `registration_parameters.angles_range` | two numbers | Minimum and maximum in-plane rotation angles, in degrees, tested during registration. |
| `registration_parameters.zoom_range` | two numbers | Minimum and maximum zoom factors allowed for each image axis. A value of `1.0` means no scaling. |
| `registration_parameters.apply_zoom` | boolean | When `true`, estimates independent scale factors along the two image axes. |
| `registration_parameters.verbose` | boolean | When `true`, generates intermediate correlation plots. |
| `reference_DAPI_folder` | path | Folder containing the fixed reference TIFF stack. |
| `template_reference_DAPI_file` | string | Glob pattern used to identify the reference TIFF file. It must match exactly one file. |
| `number_channel_reference_stack` | integer | Total number of interleaved channels in the reference stack. |
| `reference_channel` | integer | Zero-based index of the DAPI channel in the reference stack. |
| `target_DAPI_folder` | path | Folder containing the target TIFF stack that will be transformed. |
| `template_target_DAPI_file` | string | Glob pattern used to identify the target TIFF file. It must match exactly one file. |
| `number_channel_target_stack` | integer | Total number of interleaved channels in the target stack. |
| `target_channel` | integer | Zero-based index of the DAPI channel in the target stack. |
| `dest_root_folder` | path | Parent folder in which the analysis folder is created. |
| `dest_analysis_folder` | string | Name of the folder containing the registration outputs. |
| `overwrite_existing_file` | boolean | Controls whether existing output files associated with `data_tag` are removed before a new analysis. |

### Choosing registration parameters

#### `downsizing_power`

The analysis image is reduced by:

```text
binning factor = 2**downsizing_power
```

Examples:

| Value | Reduction along each image axis |
|---:|---:|
| `0` | no reduction |
| `1` | divide by 2 |
| `2` | divide by 4 |

Larger values make parameter estimation faster, but excessive downsizing may remove useful nuclear structure and reduce registration accuracy.

#### `angles_range`

Keep the angle interval as narrow as justified by the experiment. A very wide interval increases computation time and may increase the risk of a false match.

Example:

```yaml
angles_range: [-20, 20]
```

tests rotations between approximately -20° and +20°.

#### `apply_zoom`

Use zoom correction only when the two acquisitions are expected to differ in magnification or global scaling.

```yaml
apply_zoom: true
zoom_range: [0.95, 1.05]
```

A zoom factor greater than `1.0` enlarges the target image along that axis. A factor below `1.0` shrinks it.

Zoom correction is global. It cannot correct local deformation.

#### `overwrite_existing_file`

Use this setting carefully.

```yaml
overwrite_existing_file: true
```

removes existing output files whose names contain the current `data_tag`.

```yaml
overwrite_existing_file: false
```

keeps existing files. When exactly one matching registration-parameter JSON file is found, the saved registration parameters are loaded.

---

## Running the script

### From a terminal

The argument passed with `-F` must be the path to the YAML file, not a folder.

```bash
python Image_registration.py -F /path/to/Config_registration.yml
```

The long form is:

```bash
python Image_registration.py --config-file /path/to/Config_registration.yml
```

This README assumes that the parser and constructor consistently use the name `config_file`.



---

## Runtime options

The current main block calls:

```python
align.run_registration(
    order=1,
    downsize=True,
    full_3D=False,
)
```

### `order`

Interpolation order used when transforming the image.

- `order=1` is suitable for fluorescence-intensity images.
- `order=0` should be used when applying a saved transformation to segmentation masks or discrete label images, because it avoids creating intermediate label values.

### `downsize`

When `True`, rotation and translation are estimated from binned images for faster processing. The final transformation is still applied at the original image resolution.

### `full_3D`

When `True`, the script also estimates a global Z displacement and saves an XZ registration diagnostic.

When `False`, only the XY registration parameters are estimated. The saved `z_shift` remains at its default value unless parameters were loaded from a previous analysis.

---

## Outputs

Outputs are written to:

```text
dest_root_folder/dest_analysis_folder
```

The script currently creates the following files.

### Registration parameters

```text
<data_tag>_registration_parameters.json
```

This JSON file contains:

```json
{
  "shift": [0, 0],
  "rotation": 0,
  "zoom": [1, 1],
  "z_shift": 0
}
```

The values represent:

- `shift`: displacement along image axes 0 and 1, in pixels of the original image;
- `rotation`: in-plane rotation, in degrees;
- `zoom`: scale factors along image axes 0 and 1;
- `z_shift`: displacement along Z, in image planes.

Users should normally apply these values through the registration methods rather than manually reproducing the sign conventions.

### XY montage

```text
MIP_registered_<data_tag>_registered.png
```

This image contains:

1. the reference DAPI maximum-intensity projection;
2. the unregistered target DAPI maximum-intensity projection;
3. the registered overlay.

### XY overlay

```text
MIP_registered_<data_tag>_registered_montage.png
```

The recommended color convention is:

- **red:** reference DAPI;
- **green:** registered target DAPI;
- **yellow:** overlap between reference and registered target.

Yellow is not a separate signal. It is produced where red and green structures overlap.

### XZ diagnostic

When `full_3D=True`, the script also saves:

```text
MIP_registered_<data_tag>_registered_montages_XZ.png
```

This figure shows several orthogonal projections used to inspect the Z registration.

### Registered TIFF stack

The script calculates a final registered target stack, but saving it is disabled in the current code because the corresponding lines are commented out.

To save it, enable the relevant lines in `run_registration()`:

```python
saving_filename = f"{Path(path_to_align[0]).stem}_registered.tif"
self.save_tiff_image(
    stack_registered_final,
    saving_filename,
    rescale=False,
)
```

---

## Applying the transformation to other data

After estimating the transformation from the DAPI channel, the same transformation may be applied to another image acquired in the target experiment, provided that it uses the same image coordinate system.

For an intensity image:

```python
registered_image = align.apply_2d_registration(
    target_image,
    order=1,
    downsize=False,
)
```

For a segmentation mask or label image:

```python
registered_mask = align.apply_2d_registration(
    target_mask,
    order=0,
    downsize=False,
)
```

Use nearest-neighbour interpolation (`order=0`) for masks and labels.

For a complete stack:

```python
registered_stack = align.apply_3d_registration(
    target_stack,
    order=1,
    downsize=False,
)
```

Coordinate tables, HiM traces, or detected RNA positions require an explicit coordinate-transformation procedure. The current script primarily applies transformations to image arrays and does not provide a dedicated function for transforming point-coordinate tables.

---