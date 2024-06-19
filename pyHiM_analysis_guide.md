# Guide to process HiM data

The basic steps are:
- link raw images from an experiment into a single folder
- deinterleave images
- deconvolve images
- sort deconvolved images into ROI folders
- analyze using pyHiM
- benchmark analysis ROI per ROI.
- choose tracing method (mask, KDtree)
- analyze and filter traces for each ROI.
- label images and sort traces for each label [optional]
- merge traces from different ROIs 
- convert trace file to PWD and proximity maps.

## Installations

### pyHiM

Follow installation instructions [here](https://pyhim.readthedocs.io/en/latest/).

### pyHiM_tools

You need to:
- create Repository folder in your home directory
- clone the repository
- add the folder with the scripts to your PATH variable

```
$ mkdir $HOME/Repositories && cd $HOME/Repositories
$ git clone git@github.com:pyHi-M/pyHiM_tools.git
$ echo 'export PATH="$PATH:$HOME/Repositories/pyHiM_tools/tools"' > ~/.bashrc
$ bash
```

If the cloning does not work, check with marcelo if you are in the repository owners, and if you set the right SSH key. For this latter see [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

### lab_scripts

You need to:
- create Repository folder in your home directory
- clone the repository
- add the folder with the scripts to your PATH variable

```
$ mkdir $HOME/Repositories && cd $HOME/Repositories
$ git clone git@github.com:marcnol/lab_scripts.git
$ echo 'export PATH="$PATH:$HOME/Repositories/lab_scripts/imaging/deconwolf"' > ~/.bashrc
$ bash
```

### cellpose

Please check guide [here](https://github.com/MouseLand/cellpose) for the most updated instructions!

The most convenient way is using a conda environment. 

#### Installation Instructions with conda

If you have an older cellpose environment you can remove it with conda env remove -n cellpose before creating a new one.

If you are using a GPU, make sure its drivers and the cuda libraries are correctly installed.

Install an Anaconda distribution of Python. Note you might need to use an anaconda prompt if you did not add anaconda to the path.
Open an anaconda prompt / command prompt which has conda for python 3 in the path
Create a new environment with conda create --name cellpose python=3.8. We recommend python 3.8, but python 3.9 and 3.10 will likely work as well.
To activate this new environment, run conda activate cellpose
To install the minimal version of cellpose, run python -m pip install cellpose.
To install cellpose and the GUI, run python -m pip install cellpose[gui]. If you're on a zsh server, you may need to use ' ' around the cellpose[gui] call: python -m pip install 'cellpose[gui]'.
To upgrade cellpose (package here), run the following in the environment:

python -m pip install cellpose --upgrade
Note you will always have to run conda activate cellpose before you run cellpose. If you want to run jupyter notebooks in this environment, then also python -m pip install notebook and python -m pip install matplotlib.

You can also try to install cellpose and the GUI dependencies from your base environment using the command

python -m pip install cellpose[gui]
If you have issues with installation, see the docs for more details. You can also use the cellpose environment file included in the repository and create a cellpose environment with conda env create -f environment.yml which may solve certain dependency issues.

If these suggestions fail, open an issue.


## Pre-processing

### link raw images from an experiment into a single folder

ASK OLIVIER


### deinterleave images

For this use the `deinterleave_channels.py` function from `pyHiM_tools`.

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

### deconvolve images



### sort deconvolved images into ROI folders

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

This will sort out files within the folder `deconvolved`. If `output` is not provided, the folder will be created in the current folder. 


## pyHiM analysis

### Preparing BASH scripts to run pyHiM in batch

This script will prepare a BASH script to run pyHiM sequentially or in parallel in each ROI folder.

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


### Run projection and registrations

To run this modules on a single folder,

```
$ cd 001_ROI
$ conda activate pyHiM39
$ pyHiM.py -C project,register_global,register_local
```

To make a BASH script to run in multiple folders at once

```
$ runHiM_cluster.py -F folder_with_ROIs --bash -C project,register_global,register_local
```

then run the bash script
```
$ bash joblist_folder_with_ROIs.bash
```


### Run mask_3D

To run this modules on a single folder,

```
$ cd 001_ROI
$ conda activate pyHiM39
$ pyHiM.py -C mask_3d
```

To make a BASH script to run in multiple folders at once

```
$ runHiM_cluster.py -F folder_with_ROIs --bash -C mask_3d
```

then run the bash script
```
$ bash joblist_folder_with_ROIs.bash
```


### Run DAPI segmentations using cellpose [optional]


#### run segmentation

For this you need to run the `mask_cellpose.py` script that lives in `pyHiM`. Run in lopevi or lifou where 2-3 GPUs are available to speed up segmentation. The default parameters are set for DAPI segmentations.

See instructions here:

[mask_cellpose](https://pyhim.readthedocs.io/en/latest/user_guide/modules/identification/mask_cellpose.html)


Example:
```
$ ls *DAPI*ch00.tif | mask_cellpose --input scan_001_DAPI_001_ROI_converted_decon_ch00.tif
```


[Debugging] Make sure the `src/postProcessing` folder is in your PATH. Otherwise, run:

```
$ echo 'export PATH="$PATH:$HOME/Repositories/pyHiM/src/postProcessing"' > ~/.bashrc
$ bash
```

### Run localize_3d, filter_localizations, register_localizations

To run this modules on a single folder,

```
$ cd 001_ROI
$ conda activate pyHiM39
$ pyHiM.py -C localize_3d,filter_localizations,register_localizations
```

To make a BASH script to run in multiple folders at once

``
$ runHiM_cluster.py -F folder_with_ROIs --bash -C localize_3d,filter_localizations,register_localizations
```

then run the bash script
```
$ bash joblist_folder_with_ROIs.bash

### Run build_trace

To run this modules on a single folder,

```
$ cd 001_ROI
$ conda activate pyHiM39
$ pyHiM.py -C build_traces
```

To make a BASH script to run in multiple folders at once

``
$ runHiM_cluster.py -F folder_with_ROIs --bash -C build_traces
```

then run the bash script
```
$ bash joblist_folder_with_ROIs.bash

### benchmark analysis ROI per ROI.

You need to check
- projections: do all barcodes worked?, do fiducial marks look normal?
- registration_global: did registrations worked? what cycles presented problems? do you see incomplete registration dur to deformation ? Check that the number of blocks worked fine.
- registration_local: do registrations look normal? For the cycles that displayed deformation, do they show better corrections? How do the shift matrices look? Are their homogeneous? Do you see very large correction values (>1) ?
- mask_3d: check how well the DAPI and mask segmentations worked. See Jupyter lab to compare segmentation and deconvolved image in 3D.
- localize_3D: explore masks and localizations. Do you see over or under segmentation? See Jupyter lab to compare segmentation and deconvolved image in 3D. Check the statistics of localizations. Find a flux threshold that would cut out noisy localizations.
- builds_traces: check performance of each of the tracing methods you used. For this look at the XY XZ YZ trace representations overlayed with the masks.
- build_matrix: check the N-matrix for barcodes with low number of detections. For these, verify if it is due to labeling efficiency, localization efficiency, or problems with drift correction. Check efficiency of barcode detections. Check for XYZ ranges that may need to be removed (background on bottom or top of slide/coverslip surface? dirt spots?). These regions should be removed in post-processing.

### choose tracing method (mask, KDtree)

Based on the analysis above you should choose one of the tracing methods and perform post-processing using it.

## Post-processing

### analyze and filter traces for each ROI.

See guides here:

[trace_analyser](https://pyhim.readthedocs.io/en/latest/user_guide/modules/building_traces/trace_analyser.html)
[trace_filter](https://pyhim.readthedocs.io/en/latest/user_guide/modules/building_traces/trace_filter.html)

### label images and sort traces for each label [optional]

See guides here:

[mask_manual](https://pyhim.readthedocs.io/en/latest/user_guide/modules/identification/mask_manual.html)

[trace_assign_mask](https://pyhim.readthedocs.io/en/latest/user_guide/modules/building_traces/trace_assign_mask.html)


### merge traces from different ROIs 

See guides here:

[trace_merge](https://pyhim.readthedocs.io/en/latest/user_guide/modules/building_traces/trace_merge.html)


### convert trace file to PWD and proximity maps.

See guides here:

[build_matrices](https://pyhim.readthedocs.io/en/latest/user_guide/modules/building_traces/build_matrices.html)

### 3D trace plots 

See guides here:

[trace_plot](https://pyhim.readthedocs.io/en/latest/user_guide/modules/building_traces/trace_plot.html)
