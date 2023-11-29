import logging
import os.path as op
import re

import cv2 as cv
import nibabel as nib
import numpy as np
from alive_progress import alive_it
from paths import glob_sub_dir, root_dir
from skimage.feature import canny
from skimage.morphology import (
    convex_hull_image,
    dilation,
    disk,
    square,
    white_tophat,
)
from util.io import load_nifti, save_nifti
from util.roi import binarize_img

sub_dirs = glob_sub_dir(root_dir)
if sub_dirs is None:
    raise FileNotFoundError(
        f"No files found in {root_dir} matching the folder structure"
    )

for sub_dir in alive_it(sub_dirs):
    logging.basicConfig(level=logging.INFO)
    sub, ses, run = re.findall(r"([0-9]+)", sub_dir)
    petra_path = op.join(sub_dir, "petra_.nii.gz")
    cylinder_mask_path = op.join(sub_dir, "cylinder_ROI.nii.gz")

    if not op.exists(petra_path):
        logging.warning(
            f"Did not find {petra_path} for sub-{sub}, ses-{ses}, "
            + f"run-{run}.\nWill skip the subject"
        )
        continue

    if not op.exists(cylinder_mask_path):
        logging.warning(
            f"Did not find {cylinder_mask_path} for sub-{sub}, ses-{ses}, "
            + "run-{run}.\nWill skip the subject"
        )
        continue

    petra, petra_img = load_nifti(petra_path)
    _, mask_img = load_nifti(cylinder_mask_path)
    _, layers_img = load_nifti(op.join(sub_dir, "layers.nii.gz"))
    _, head_mask_img = load_nifti(op.join(sub_dir, "finalmask.nii.gz"))

    logging.info(f"Number of Voxel pre-dilation: {np.sum(mask_img)}")

    mask_img_dil = np.array(
        [
            dilation(mask, square(3)) if np.any(mask) else mask
            for mask in mask_img
        ]
    )
    logging.info(f"Number of Voxel post-dilation: {np.sum(mask_img_dil)}")

    layers_img_bin = binarize_img(layers_img, threshold=3)
    head_mask = binarize_img(head_mask_img, threshold=1)
    mask_img_dil_cut = mask_img_dil * ~layers_img_bin * ~head_mask
    masked_petra = petra_img * mask_img_dil_cut

    logging.info(
        f"N voxels cut from mask: {np.sum(mask_img_dil - mask_img_dil_cut)}"
    )

    masked_denoised_petra = np.array(
        [white_tophat(img, disk(3)) for img in masked_petra]
    )

    sigma = 1

    edges = np.array(
        [canny(img, sigma=sigma) for img in masked_denoised_petra]
    )

    if not np.any(edges):
        logging.warning(
            f"Did not find any edges for sub-{sub}, ses-{ses}, "
            + f"run-{run}.\nWill skip the subject"
        )
        continue

    chull = np.array(
        [convex_hull_image(img) if np.any(img) else img for img in edges]
    )

    save_nifti(edges, op.join(sub_dir, f"canny_sigma_{sigma}.nii.gz"), petra)
    save_nifti(chull, op.join(sub_dir, "chull.nii.gz"), petra)
