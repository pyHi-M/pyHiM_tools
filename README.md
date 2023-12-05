# pyHiM_tools

#### deinterleave_channels.py

Example:
```
$ ls *tif | deinterleave_channels.py --N_channels 2 --pipe
```

Usage:
```sh
usage: deinterleave_channels.py [-h] [--input INPUT] [--N_channels N_CHANNELS] [--pipe]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Name of input trace file.
  --N_channels N_CHANNELS
                        Number of channels in image.
  --pipe                inputs Trace file list from stdin (pipe)
```

#### extract_subVolume_TIFF.py
```sh
usage: extract_subVolume_TIFF.py [-h] [-F FILE] [-Z ZOOM]

optional arguments:
  -h, --help            show this help message and exit
  -F FILE, --file FILE  folder TIFF filename
  -Z ZOOM, --zoom ZOOM  provide zoom factor
```

### denoising

#### run_aydin.py

Example:
```
$ ls *ch00.tif | run_aydin.py --pipe
```

Usage:
```sh
usage: run_aydin.py [-h] [--input INPUT] [--dict_path DICT_PATH] [--pipe]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Name of input trace file.
  --dict_path DICT_PATH
                        Path to the dictionary of aydin parameters generated using the GUI
  --pipe                inputs Trace file list from stdin (pipe)
```

### AI segmentation
`trainStarDist.py`


#### run_cellpose.py

Example:
```
$ ls *_denoised.tif | run_cellpose --pipe
```

Usage:
```sh
usage: run_cellpose.py [-h] [--input INPUT] [--cellprob CELLPROB] [--flow FLOW] [--stitch STITCH] [--diam DIAM] [--pipe]

optional arguments:
  -h, --help           show this help message and exit
  --input INPUT        Name of input trace file.
  --cellprob CELLPROB  cellprob threshold. Default = -8.
  --flow FLOW          flow threshold. Default = 10.
  --stitch STITCH      stitch threshold. Default = 0.1.
  --diam DIAM          diameter. Default = 50.
  --pipe               inputs Trace file list from stdin (pipe)
```

### post-processing 3D masks

#### mask_analyze.py
```sh
usage: mask_analyze.py [-h] [--input INPUT] [--pipe]

optional arguments:
  -h, --help     show this help message and exit
  --input INPUT  Name of input trace file.
  --pipe         inputs Trace file list from stdin (pipe)
```


#### process_3D_masks.py
```sh
usage: process_3D_masks.py [-h] [--input INPUT] [--num_pixels_min NUM_PIXELS_MIN] [--save] [--convert] [--z_min Z_MIN]
                           [--z_max Z_MAX] [--y_min Y_MIN] [--y_max Y_MAX] [--x_min X_MIN] [--x_max X_MAX] [--pipe]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Name of input trace file.
  --num_pixels_min NUM_PIXELS_MIN
                        Masks with less that this number of pixels will be removed. Default = 0
  --save                Saves the processed 3D image
  --convert             Converts from NPY to TIF or vice versa depending on input
  --z_min Z_MIN         Z minimum for a localization. Default = 0
  --z_max Z_MAX         Z maximum for a localization. Default = np.inf
  --y_min Y_MIN         Y minimum for a localization. Default = 0
  --y_max Y_MAX         Y maximum for a localization. Default = np.inf
  --x_min X_MIN         X minimum for a localization. Default = 0
  --x_max X_MAX         X maximum for a localization. Default = np.inf
  --pipe                inputs Trace file list from stdin (pipe)
```


### showing PDB files in pymol
`loadDir.pml`
`loadDir.py`

### post-processing matrices

#### 1. to get the genomic distance map

This script loads
1. a BED file with the coordinates of barcodes. 
2. a CSV file with unique barcodes

**The names of the barcodes in the BED file HAVE TO BE the same as in the CSV file.**

from this it creates a matrix of genomic distances that are exported as PNG and as NPY

Example:

```sh
$ get_genomic_distance_map.py --barcodes_file_path uniqueBarcodes.ecsv --bed_file_path 3R_All_barcodes.bed
```

```
usage: get_genomic_distance_map.py [-h] [--barcodes_file_path BARCODES_FILE_PATH] [--bed_file_path BED_FILE_PATH]
                                   [--file_output FILE_OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  --barcodes_file_path BARCODES_FILE_PATH
                        Name of input barcode list, in csv format.
  --bed_file_path BED_FILE_PATH
                        Name of input barcode coordinates file, in bed format.
  --file_output FILE_OUTPUT
                        Name of output files.
```

#### 2. to normalize the proximity map by genomic distance

This script loads
    -a list of PWD maps in NPY format
    -a genomic distance map based on a list of barcodes, generated using `get_barcode_normalisation_map.py`
    -a CSV file with uniquebarcodes

from this it:
    - calculates the proximity frequency map
    - normalizes it by the number of times two barcodes are found in a trace
    - calculates the power law decay of proximity with genomic distance
    - constructs the expected proximity map
    - gets the observed/expected proximity map

Example:
```sh
$ normalize_PWD_map.py --input merged_traces_Matrix_PWDscMatrix.npy --genomic_distance_map genomic_distance_map.npy --uniqueBarcodes uniqueBarcodes.ecsv
```

```
usage: normalize_PWD_map.py [-h] [--genomic_distance_map GENOMIC_DISTANCE_MAP] [--input INPUT]
                            [--file_output FILE_OUTPUT] [--proximity_threshold PROXIMITY_THRESHOLD]
                            [-U UNIQUEBARCODES]

optional arguments:
  -h, --help            show this help message and exit
  --genomic_distance_map GENOMIC_DISTANCE_MAP
                        Name of genomic distance matrix, in NPY format.
  --input INPUT         Name of input PWD maps, in NPY format.
  --file_output FILE_OUTPUT
                        Name of output files.
  --proximity_threshold PROXIMITY_THRESHOLD
                        proximity threshold in um
  -U UNIQUEBARCODES, --uniqueBarcodes UNIQUEBARCODES
                        csv file with list of unique barcodes
```

### compare multiple NPY matrices using pearson correlation

The script `` can compare two matrices. But it is annoying having to call it several times to compare multiple datasets. For this I developed ``.
The datasets to be compared are sent by piping. The dataset that all datasets are compared against are sent by the argument --input. See example below:

```
usage: run_multiple_correlations.py [-h] [-I INPUT]

optional arguments:
  -h, --help            show this help message and exit
  -I INPUT, --input INPUT
                        Input file against which all files should be compared to
```

Example
```
$ ls Trace*PWDscMatrix.npy | python -m run_multiple_correlations.py --input merged_traces_Matrix_PWDscMatrix.npy
```


