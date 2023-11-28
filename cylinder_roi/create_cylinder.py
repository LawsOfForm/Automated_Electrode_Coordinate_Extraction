import logging
import os.path as op
import re

import nibabel as nib
import numpy as np
from alive_progress import alive_it
from paths import glob_sub_dir, root_dir
from util.io import read_mricoords, save_nifti
from util.roi import cylinder
from util.transform import (
    fill_holes,
    get_normal_component,
    get_rotation_matrix,
    project_onto_plane,
    rotate_img_obj,
    img_insert_value_at_ind,
)


# TODO: add multiple cylinders

sub_dirs = glob_sub_dir(root_dir)

if sub_dirs is None:
    raise FileNotFoundError("No sub-directories found.")

for sub_dir in alive_it(sub_dirs):
    # logging.basicConfig(level=logging.INFO)
    cylinder_mask_path = op.join(sub_dir, "cylinder_ROI.nii.gz")
    cylinder_mask_plus_plug = op.join(sub_dir, "cylinder_plug_plug_ROI.nii.gz")
    final_mask_path = op.join(sub_dir, "finalmask.nii.gz")

    sub, ses, run = re.findall(r"([0-9]+)", sub_dir)

    if not op.exists(final_mask_path):
        continue
    if op.exists(cylinder_mask_path) and op.exists(cylinder_mask_plus_plug):
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
            get_normal_component(mricoords[i : i + 3]) for i in (first_non_centre_ind)
        ]
        point_on_plane = mricoords[first_non_centre_ind]
        centres = [
            project_onto_plane(c, n, p)
            for c, n, p in zip(centres, normal_components, point_on_plane)
        ]

    height = 2
    radius = 10
    empty_img = np.zeros(nifti.shape)

    cylinder_masks = [cylinder(nifti, c, radius, height) for c in centres]

    rotation_matrices = [
        get_rotation_matrix(np.array([0, 0, 1]), n) for n in normal_components
    ]

    rotated_cylinder_inds = np.array(
        [
            rotate_img_obj(cylinder_mask, rotation_matrix, c)
            for cylinder_mask, rotation_matrix, c in zip(
                cylinder_masks, rotation_matrices, centres
            )
        ]
    )

    emtpy_img = np.zeros((nifti.shape))

    rotated_cylinder = img_insert_value_at_ind(
        img=empty_img,
        inds=rotated_cylinder_inds,
        value=1,
    )

    rotated_cylinder = fill_holes(emtpy_img)

    save_nifti(img=rotated_cylinder, path=cylinder_mask_path, ref=nifti)

    # add plug

    plug_height = height + 5
    plug_radius = radius / 2

    plug_masks = [cylinder(nifti, c, plug_radius, plug_height) for c in centres]

    rotated_plugs_inds = np.array(
        [
            rotate_img_obj(plug_mask, rotation_matrix, c)
            for plug_mask, rotation_matrix, c in zip(
                plug_masks, rotation_matrices, centres
            )
        ]
    )

    rotated_cyl_plus_plugs = img_insert_value_at_ind(
        img=rotated_cylinder,
        inds=rotated_plugs_inds,
        value=1,
    )

    rotated_cyl_plus_plugs = fill_holes(rotated_cyl_plus_plugs)

    save_nifti(img=rotated_cyl_plus_plugs, path=cylinder_mask_plus_plug, ref=nifti)
