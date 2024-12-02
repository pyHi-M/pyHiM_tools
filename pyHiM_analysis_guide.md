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

Install pyHiM locally and add paths:

```
$ mkdir $HOME/Repositories && cd $HOME/Repositories
$ git clone git@github.com:marcnol/pyHiM.git
$ echo 'export PATH="$PATH:$HOME/Repositories/pyHiM/src"' >> ~/.bashrc
$ echo 'export PATH="$PATH:$HOME/Repositories/pyHiM/src/postProcessing"' >> ~/.bashrc
$ echo 'export PATH="$PATH:$HOME/Repositories/pyHiM/src/plots"' >> ~/.bashrc
$ bash
```

Follow installation instructions [here](https://pyhim.readthedocs.io/en/latest/).

### pyHiM_tools

You need to:
- create Repository folder in your home directory
- clone the repository
- add the folder with the scripts to your PATH variable

```
$ mkdir $HOME/Repositories && cd $HOME/Repositories
$ git clone git@github.com:pyHi-M/pyHiM_tools.git
$ echo 'export PATH="$PATH:$HOME/Repositories/pyHiM_tools/tools"' >> ~/.bashrc
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
$ echo 'export PATH="$PATH:$HOME/Repositories/lab_scripts/imaging/deconwolf"' >> ~/.bashrc
$ bash
```

### cellpose

Please check guide [here](https://github.com/MouseLand/cellpose) for the most updated instructions!

The most convenient way is using a conda environment. 

For cellpose 3:
```
conda create -n cellpose3 pytorch=1.8.2 cudatoolkit=10.2 -c pytorch-lts
conda activate cellpose3
pip install cellpose
```

For 3D segmentation using the GUI, run:
```
python -m cellpose --Zstack
```

if you get an error regarding both Numpy 1.X and 2.X not being able to run at the same time like this:

```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.1 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
```

then apply this patch:

```
pip uninstall numpy
pip install "numpy<2.0"
```

If you get an error regarding `GUI ERROR: No module named 'qtpy'` then run this:

```
pip install 'cellpose[gui]'
```


#### Installation Instructions with conda

If you have an older cellpose environment you can remove it with 
```
conda env remove -n cellpose 
```
before creating a new one.

If you are using a GPU, make sure its drivers and the cuda libraries are correctly installed.

Install an Anaconda distribution of Python. Note you might need to use an anaconda prompt if you did not add anaconda to the path.

Open an anaconda prompt / command prompt which has conda for python 3 in the path

Create a new environment with 
```
conda create --name cellpose python=3.8
```

We recommend python 3.8, but python 3.9 and 3.10 will likely work as well.

To activate this new environment, run 
```
conda activate cellpose
```

Now install cellpose in the environment:
```
python -m pip install cellpose[gui]
```


If you want to run jupyter notebooks in this environment, then also 
```
python -m pip install notebook and python -m pip install matplotlib
```

## Pre-processing

### deinterleave images

For this use the `deinterleave_channels.py` function from `pyHiM_tools`.

Example if you performed an experiment and the raw data are in `/mnt/grey/DATA/rawData_2024/Experiment_X`:

Deinterleave data in situ:

```
cd /mnt/grey/DATA/rawData_2024/Experiment_X/deinterleaved
ls */*tif | deinterleave_channels.py --N_channels 2 --pipe
```

Create destination folder and move data therein:
```
mkdir /mnt/grey/DATA/ProcessedData_2024/Experiment_X/deinterleaved
mv */*ch0*.tif /mnt/grey/DATA/ProcessedData_2024/Experiment_X/deinterleaved
```

If your data has 3 channels, then adapt as follows:
```
cd /mnt/grey/DATA/rawData_2024/Experiment_X/deinterleaved
ls */*tif | deinterleave_channels.py --N_channels 3 --pipe
```

To deinterleave barcodes, you need to expand the `ls` command so it can find the tifs:
```
cd /mnt/grey/DATA/rawData_2024/Experiment_X/deinterleaved
ls */*/*tif | deinterleave_channels.py --N_channels 2 --pipe
```

Remember then to move the files once you are done:
```
mv */*/*ch0*.tif /mnt/grey/DATA/ProcessedData_2024/Experiment_X/deinterleaved
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

#### deconwolf


Finally run this script to process all tif files in a the folder holding the deinterleaved images:

```sh
$ deconvolve_dw.py --gpu --tilesize 1024
```
Note: tilesize needs to be adapted to the image size. For images acquired with the Hamamatsu sCMOS (2048x2048) a tilesize of `1024` works fine. However, for the Kinetix (3200x3200) we need to use instead `512`.

Run this command in either lifou or lopevi.

This script assumes that 
- DAPI ch00 was acquired at 400nm
- DAPI ch01 was acquired at 488nm (fiducial)
- RT ch00 was acquired at 488nm (fiducial)
- RT ch01 was acquired at 638nm

If this was not the case for your experiment, you will need to adapt the script!

You now need to create a folder to hold the deconvolved images and move them there. For this

```
mkdir /mnt/grey/DATA/ProcessedData_2024/Experiment_X/deconvolved
mv /mnt/grey/DATA/ProcessedData_2024/Experiment_X/deinterleaved/*decon*tif /mnt/grey/DATA/ProcessedData_2024/Experiment_X/deconvolved
```


#### huygens

see instructions in the `lab_script` repository

### sort deconvolved images into ROI folders

This script will sort deconvolved files into folders. The script lives in the `pyHiM_tools` repository.

TIF files from a single ROI will be linked together into a single folder.

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
cd /mnt/grey/DATA/ProcessedData_2024/Experiment_X/
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
$ runHiM_cluster.py -F /home/marcnol/grey/ProcessedData_2024/Experiment_49_David_RAMM_DNAFISH_Bantignies_proto_G1E_cells_LRKit/deinterleave_deconvolved_test -C project,register_global,register_local,mask_3d,localize_3d,filter_localizations,register_localizations --bash --parallel
```

The ``--parallel` argument will execute all ROIs simultaneously. Otherwise, they will run sequentially.

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

#### filtering 3D masks.

Sometimes, either huygens or dw or both can oversegment 3d masks. This then creates a problem when performing chromatin tracing using masking. To deal with this issue, I created a script that can be run after `mask_3d` and before `build_traces`.

The script is called `mask_filter.py` and takes one or multiple mask files and filters masks following several criteria, including:
- intensity
- minimum number of pixels
- maximum number of pixels
- position in z (i.e masks below or above thresholds will be excluded)

Of course, you need to find what are the best values for these parameters. For this, I wrote `mask_analyze.py`. 

Example:

First take one ROI within your dataset and run

```
$ mask_analyze.py --input mask.tif --output mask_analysis --intensity_image original.tif
```

the mask.tif and original.tif files correspond to the mask and the intensity image filenames.

Remember that if you are doing this on a mask obtained using pyHiM you need to shift and z-reinterpolate the intensity image first. For this, use `image_interpolate_z.py` and `image_shift.py`.

Typical full example:

```
$ cd /home/marcnol/grey/ProcessedData_2024/Experiment_75_David_DNAFISH_HiM_HRKit_G1E_20240708/000

# found out shift value
$ python -c "import json; data = json.load(open('register_global/data/shifts.json')); print(list(data['ROI:000']['mask0']))"

# I got: [-9.117377046678888, 0.24557377001056907]

# shift intensity image
$ shift_3d_image.py -F scan_002_mask0_000_ROI_converted_decon_ch01.tif --shift_x -9.1 --shift_y 0.24 --shift_z 0

# interpolate in z
$ image_interpolate.py --input scan_002_mask0_000_ROI_converted_decon_ch01_shifted.tif

# run mask_analyze
$ mask_analyze.py --input mask_3d/data/scan_002_mask0_000_ROI_converted_decon_ch01_3Dmasks.npy --intensity_image scan_002_mask0_000_ROI_converted_decon_ch01_shifted_z_reinterpolated.tif --replace_mask_file
```

This should produce plots within mask_3d/data/ that you can use to establish the threshold intensity, min_z, max_z and number of pixels per mask.

You can verify how this worked by loading into napari:
- original shifted/interpolated intensity image: scan_002_mask0_000_ROI_converted_decon_ch01_shifted_z_reinterpolated.tif
- filtered masks at: mask_3d/data/scan_002_mask0_000_ROI_converted_decon_ch01_3Dmasks.npy
- original masks at: mask_3d/data/scan_002_mask0_000_ROI_converted_decon_ch01_3Dmasks_original.npy

If you are satisfied then you can apply these filters to a series of ROI. For this, move to the folder holding the ROIs, determine the `ls` command that will list all the `intensity mask images`, and run `mask_filter`:

```
$ cd /home/marcnol/grey/ProcessedData_2024/Experiment_75_David_DNAFISH_HiM_HRKit_G1E_20240708

# list all intensity images to process:
$ ls *mask0*ch01*

# if you are happy with this list, then run mask filter on this list
$ ls *mask0*ch01* | mask_filter.py --pipe --pyHiM --min_intensity 2000 --min_z 5 --max_z 20  --replace_mask_file

# or run this for several ROIs
$ cd /home/marcnol/grey/ProcessedData_2024/Experiment_75_David_DNAFISH_HiM_HRKit_G1E_20240708
$ ls */*mask*ch01.tif | mask_filter.py --pipe --pyHiM --min_intensity 2000 --min_z 5 --max_z 20  --replace_mask_file
```   


### Run DAPI segmentations using cellpose [optional]


#### run segmentation

For this you need to run the `mask_cellpose.py` script that lives in `pyHiM` (**make sure you cloned pyHiM locally and that you linked the folder as indicated at the top of these instructions file**). Run in lopevi or lifou where 2-3 GPUs are available to speed up segmentation. The default parameters are set for DAPI segmentations.

See instructions here:

[mask_cellpose](https://pyhim.readthedocs.io/en/latest/user_guide/modules/identification/mask_cellpose.html)


Example:
```
$ mask_cellpose --input scan_001_DAPI_001_ROI_converted_decon_ch00.tif --gpu
```


[Debugging] Make sure the `src/postProcessing` folder is in your PATH. Otherwise, run:

```
$ echo 'export PATH="$PATH:$HOME/Repositories/pyHiM/src/postProcessing"' >> ~/.bashrc
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

```
$ runHiM_cluster.py -F folder_with_ROIs --bash -C localize_3d,filter_localizations,register_localizations
```

then run the bash script
```
$ bash joblist_folder_with_ROIs.bash
```

### Run build_trace

To run this modules on a single folder,

```
$ cd 001_ROI
$ conda activate pyHiM39
$ pyHiM.py -C build_traces
```

To make a BASH script to run in multiple folders at once

```
$ runHiM_cluster.py -F folder_with_ROIs --bash -C build_traces
```

then run the bash script
```
$ bash joblist_folder_with_ROIs.bash
```

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
