import logging
import os.path as op
import re

import nibabel as nib
import numpy as np
from alive_progress import alive_it
from paths import glob_sub_dir, root_dir
from util.io import read_mricoords
from util.roi import cylinder
from util.transform import (
    fill_holes,
    get_normal_component,
    get_rotation_matrix,
    project_onto_plane,
    rotate_img_obj,
)

sub_dirs = glob_sub_dir(root_dir)

if sub_dirs is None:
    raise FileNotFoundError("No sub-directories found.")

for sub_dir in alive_it(sub_dirs):
    # logging.basicConfig(level=logging.INFO)
    cylinder_mask_path = op.join(sub_dir, "cylinder_ROI.nii.gz")
    final_mask_path = op.join(sub_dir, "finalmask.nii.gz")

    sub, ses, run = re.findall(r"([0-9]+)", sub_dir)

    if not op.exists(final_mask_path):
        continue
    if op.exists(cylinder_mask_path):
        continue

    nifti = nib.load(final_mask_path)

    mricoords = read_mricoords(op.join(sub_dir, "mricoords_1.mat"))

    n_coords = mricoords.shape[0]
    n_electrodes = 4
    coords_per_electrode = n_coords / n_electrodes

    if n_coords != 12 or n_coords != 16:
        logging.warning(
            f"sub-{sub} has {mricoords.shape[0]} electrode coordinates "
            + f"in ses-{ses}, run-{run}. Expected 12 or 16.\n"
            "Will skip subject."
        )
        continue

    logging.info(f"Creating cylinder ROI for sub-{sub}, ses-{ses}, run-{run}")

    centres_ind = np.arange(
        0,
        n_coords,
        coords_per_electrode,
    )
    centres = mricoords[centres_ind]

    if n_coords == 12:
        normal_components = [
            get_normal_component(mricoords[i : i + 3]) for i in centres_ind
        ]
    else:
        first_non_centre_ind = centres_ind + 1
        normal_components = [
            get_normal_component(mricoords[i : i + 3])
            for i in (first_non_centre_ind)
        ]
        point_on_plane = mricoords[first_non_centre_ind]
        centres = [
            project_onto_plane(c, n, p)
            for c, n, p in zip(centres, normal_components, point_on_plane)
        ]

    height = 5
    radius = 10
    empty_img = np.zeros(nifti.shape)

    cylinder_masks = [cylinder(nifti, c, radius, height) for c in centres]

    rotation_matrices = [
        get_rotation_matrix(np.array([0, 0, 1]), n) for n in normal_components
    ]

    rotated_cylinder_inds = [
        rotate_img_obj(cylinder_mask, rotation_matrix, c)
        for cylinder_mask, rotation_matrix, c in zip(
            cylinder_masks, rotation_matrices, centres
        )
    ]

    emtpy_img = np.zeros((nifti.shape))

    for rotated_cylinder_ind in rotated_cylinder_inds:
        emtpy_img[
            rotated_cylinder_ind[:, 0],
            rotated_cylinder_ind[:, 1],
            rotated_cylinder_ind[:, 2],
        ] = 1

    rotated_cylinder = fill_holes(emtpy_img)

    new_img = nib.Nifti1Image(rotated_cylinder, nifti.affine, nifti.header)

    nib.save(
        new_img,
        cylinder_mask_path,
    )
