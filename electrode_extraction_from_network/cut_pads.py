import os.path as op

import numpy as np
#import simpleitk as sitk
import SimpleITK as sitk
from alive_progress import alive_it
from paths import root_dir
import glob
from paths_funcs import glob_sub_dir
from util.io import load_nifti, save_nifti

DILATION_RADIUS = 6
sub_dirs = glob_sub_dir(root_dir)

if sub_dirs is None:
    raise FileNotFoundError("No subdirectories found in root directory.")

for sub_dir in sub_dirs:
    print(sub_dir)
    petra_path = op.join(sub_dir, "petra_.nii.gz")
    finalmask_path = op.join(sub_dir, "finalmask.nii.gz")
    layers_path = op.join(sub_dir, "layers_binarized.nii.gz")

     
    if not op.exists(petra_path):
        continue

    petra, petra_img = load_nifti(petra_path)
    _, finalmask_img = load_nifti(finalmask_path)
    _, layers_img = load_nifti(layers_path)

    finalmask_img = np.where(finalmask_img == 1, 1, 0)
    combined_mask = np.where((finalmask_img + layers_img) == 1, 1, 0)

    combined_mask = sitk.GetImageFromArray(combined_mask)
    dilation = sitk.BinaryDilateImageFilter()
    dilation.SetKernelRadius(DILATION_RADIUS)
    dilation.SetForegroundValue(1)
    combined_mask = dilation.Execute(combined_mask)

    combined_mask = sitk.GetArrayFromImage(combined_mask)

    cut_pads = np.where(combined_mask == 0, 0, petra_img)

    save_nifti(
        img=cut_pads, path=op.join(sub_dir, f'petra_cut_pads_dil_{DILATION_RADIUS}.nii.gz'), ref=petra
    )
