# pyHiM_tools

#### deinterleave_channels.py
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

