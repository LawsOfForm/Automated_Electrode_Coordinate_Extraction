import os.path as op
import re

import numpy as np
from alive_progress import alive_it
from paths import root_dir
from paths_funcs import glob_sub_dir
from util.divide_mask import (
    fast_divide_mask,
    recursive_flatten,
    slow_divide_mask,
)
from util.io import load_nifti, save_nifti

MAX_ELEMENT_SIZE = 4_000

sub_dirs = glob_sub_dir(root_dir)

if sub_dirs is None:
    raise FileNotFoundError("No sub-directories found.")

for sub_dir in alive_it(sub_dirs):
    cylinder_masks_path = op.join(sub_dir, "cylinder_plus_plug_ROI.nii.gz")

    if not op.exists(cylinder_masks_path):
        continue

    sub, ses, run = re.findall(r"(sub-[0-9]+|ses-[0-9]+|run-[0-9]+)", sub_dir)
    print(sub, ses, run)
    mask_nifti, mask = load_nifti(cylinder_masks_path)

    masks = fast_divide_mask(
        mask
    )  # BUG: sometimes returnes a list of lists, fix with recursive_flatten
    masks = recursive_flatten(masks)

    for i, m in enumerate(masks):
        if np.sum(m) < MAX_ELEMENT_SIZE:
            sep_mask_path = op.join(sub_dir, f"mask_{i}.nii.gz")
            save_nifti(m, sep_mask_path, mask_nifti)
            continue

        separate_masks = slow_divide_mask(m, MAX_ELEMENT_SIZE)
        for j, separate_mask in enumerate(separate_masks):
            sub_sep_mask_path = op.join(sub_dir, f"mask_{i}_{j}.nii.gz")
            save_nifti(separate_mask, sub_sep_mask_path, mask_nifti)
