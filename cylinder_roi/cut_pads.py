import os.path as op

import numpy as np
from paths import root_dir
from paths_funcs import glob_sub_dir
from util.io import load_nifti, save_nifti

sub_dirs = glob_sub_dir(root_dir)

if sub_dirs is None:
    raise FileNotFoundError("No subdirectories found in root directory.")

for sub_dir in sub_dirs:
    petra_path = op.join(sub_dir, "petra_.nii.gz")
    finalmask_path = op.join(sub_dir, "finalmask.nii.gz")
    layers_path = op.join(sub_dir, "layers_binarized.nii.gz")

    if not op.exists(petra_path):
        continue

    petra, petra_img = load_nifti(petra_path)
    _, finalmask_img = load_nifti(finalmask_path)
    _, layers_img = load_nifti(layers_path)

    combined_maks = np.where((finalmask_img + layers_img) > 0, 1, 0)

    cut_pads = np.where(combined_maks == 0, 0, petra_img)

    save_nifti(cut_pads, petra, op.join(sub_dir, "petra_cut_pads.nii.gz"))
