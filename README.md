# pyHiM_tools

## preprocessing raw data

### Sorting deconvolved files into ROI folders

This script will sort deconvolved files into folders. TIF files from a single ROI will be linked together into a single folder.

The `parameters.json` file is expected to leave in the `--input` folder. This file will be linked to each ROI folder.

Usage:
```
$ preprocess_HiM_run.py [-h] [--input [INPUT]] [--output [OUTPUT]]

Organize files into ROI folders based on filename pattern and create symbolic links.

optional arguments:
  -h, --help         show this help message and exit
  --input [INPUT]    Path to the input directory containing files to organize. Default is the current directory.
  --output [OUTPUT]  Path to the output directory. Default is the current directory.
```

Example:
```
$ preprocess_HiM_run.py --input deconvolved
```

This will sort out files within the folder `deconvolved`. As `output` is not provided, the folder will be created in the current folder. 


### Preparing BASH / SLURM scripts to run pyHiM in batch mode

This script will prepare a BASH or SLURM script to run pyHiM sequentially or in parallel in each ROI folder.

Use the argument `--bash` to produce a bash script. 

Use the argument `--parallel` to execute all ROIs as different processes in batch. Otherwise, pyHiM will be executed in a sequential mode. Beware of what modules you run in parallel and which in sequential (ie. we do not recommend running `localize_3d` or `register_local` in parallel mode).

Usage:
```
$ runHiM_cluster.py [-h] [-F DATAFOLDER] [-S SINGLEDATASET] [-A ACCOUNT] [-P PARTITION] [-N NCPU] [--memPerCPU MEMPERCPU]
                         [--nodelist NODELIST] [-T1 NTASKSNODE] [-T2 NTASKSCPU] [-C CMD] [--threads THREADS] [--srun] [--sbatch] [--bash]
                         [--parallel]

optional arguments:
  -h, --help            show this help message and exit
  -F DATAFOLDER, --dataFolder DATAFOLDER
                        Folder with data. Default: ~/scratch
  -S SINGLEDATASET, --singleDataset SINGLEDATASET
                        Folder for single Dataset.
  -A ACCOUNT, --account ACCOUNT
                        Provide your account name. Default: episcope.
  -P PARTITION, --partition PARTITION
                        Provide partition name. Default: tests
  -N NCPU, --nCPU NCPU  Number of CPUs/Task
  --memPerCPU MEMPERCPU
                        Memory required per allocated CPU in Mb
  --nodelist NODELIST   Specific host names to include in job allocation.
  -T1 NTASKSNODE, --nTasksNode NTASKSNODE
                        Number of tasks per node.
  -T2 NTASKSCPU, --nTasksCPU NTASKSCPU
                        Number of tasks per CPU
  -C CMD, --cmd CMD     Comma-separated list of routines to run: project register_global register_local mask_2d localize_2d mask_3d
                        localize_3d filter_localizations register_localizations build_traces build_matrix
  --threads THREADS     Number of threads for parallel mode. None: sequential execution
  --srun                Runs using srun
  --sbatch              Runs using sbatch
  --bash                Runs using bash
  --parallel            Runs all processes in parallel
```

Example:
```
$ runHiM_cluster.py -F /home/marcnol/grey/ProcessedData_2024/Experiment_49_David_RAMM_DNAFISH_Bantignies_proto_G1E_cells_LRKit/deinterleave_deconvolved_test/ --bash
```
Once the script is run, a bash file will be produced in the current folder. 

Run this bash file to execute pyHiM. Remember to activate the environment first (e.g. `$ conda activate pyHiM39`):
```
$ bash joblist_deinterleave_deconvolved_test.bash
```

The bash script looks as follows:

```
#!/bin/bash

# dataset: 000_completePipeline
pyHiM.py -F /home/marcnol/grey/ProcessedData_2024/Experiment_49_David_RAMM_DNAFISH_Bantignies_proto_G1E_cells_LRKit/deinterleave_deconvolved_te
st/000

# dataset: 001_completePipeline
pyHiM.py -F /home/marcnol/grey/ProcessedData_2024/Experiment_49_David_RAMM_DNAFISH_Bantignies_proto_G1E_cells_LRKit/deinterleave_deconvolved_te
st/001

# dataset: 002_completePipeline
pyHiM.py -F /home/marcnol/grey/ProcessedData_2024/Experiment_49_David_RAMM_DNAFISH_Bantignies_proto_G1E_cells_LRKit/deinterleave_deconvolved_te
st/002
```



### deinterleave_channels.py

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
## AI segmentation


### run_cellpose.py

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
### denoising using run_aydin.py

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

## Handling TIF files and 3D masks

### extract_subVolume_TIFF.py
```sh
usage: extract_subVolume_TIFF.py [-h] [-F FILE] [-Z ZOOM]

optional arguments:
  -h, --help            show this help message and exit
  -F FILE, --file FILE  folder TIFF filename
  -Z ZOOM, --zoom ZOOM  provide zoom factor
```

### mask_analyze.py
```sh
usage: mask_analyze.py [-h] [--input INPUT] [--pipe]

optional arguments:
  -h, --help     show this help message and exit
  --input INPUT  Name of input trace file.
  --pipe         inputs Trace file list from stdin (pipe)
```


### process_3D_masks.py
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

## trace representation

### showing PDB files in pymol
`loadDir.pml`
`loadDir.py`

## post-processing HiM matrices

### 1. to get the genomic distance map

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

### 2. to normalize the proximity map by genomic distance

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


