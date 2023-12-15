import logging
import os.path as op
import re

import nibabel as nib
import numpy as np
from alive_progress import alive_it
from paths import root_dir
from paths_funcs import glob_sub_dir
from util.io import read_mricoords, save_nifti
from util.roi import centroid, cylinder
from util.transform import (
    fill_holes,
    get_normal_component,
    get_rotation_matrix,
    img_insert_value_at_ind,
    project_onto_plane,
    rotate_img_obj,
)

sub_dirs = glob_sub_dir(root_dir)

if sub_dirs is None:
    raise FileNotFoundError("No sub-directories found.")

# Use this filter if you want to have only one subject with specific session and run
# sub_dirs = [sub_dir for sub_dir in sub_dirs if "010" in sub_dir if 'ses-2' in sub_dir if 'run-01' in sub_dir]


for sub_dir in alive_it(sub_dirs):
    # logging.basicConfig(level=logging.INFO)
    cylinder_mask_path = op.join(sub_dir, "cylinder_ROI.nii.gz")
    cylinder_mask_plus_plug = op.join(sub_dir, "cylinder_plus_plug_ROI.nii.gz")
    final_mask_path = op.join(sub_dir, "finalmask.nii.gz")

    sub, ses, run = re.findall(r"(sub-[0-9]+|ses-[0-9]+|run-[0-9]+)", sub_dir)

    if not op.exists(final_mask_path):
        continue
    # if op.exists(cylinder_mask_path) and op.exists(cylinder_mask_plus_plug):
    #    continue

    nifti = nib.load(final_mask_path)

    mricoords = read_mricoords(op.join(sub_dir, "mricoords_1.mat"))

    n_coords = mricoords.shape[0]
    n_electrodes = 4
    coords_per_electrode = n_coords / n_electrodes

    if not n_coords == 24:
        logging.warning(
            f"{sub} has {mricoords.shape[0]} electrode coordinates "
            + f"in {ses}, {run}. Expected 24.\n"
            + "Will skip subject. \n"
        )
        continue

    logging.info(f"Creating cylinder ROI for {sub}, {ses}, {run}")

    mid = centroid(mricoords, n_electrodes)

    np.savetxt(op.join(sub_dir, "mid.txt"), mid, delimiter=",", fmt="%i")

    centres_ind = np.arange(0, n_coords, coords_per_electrode, dtype="int8")
    centres = mid

    first_non_centre_ind = centres_ind + 1
    normal_components = [
        get_normal_component(mricoords[i : i + 5])
        for i in (first_non_centre_ind)
    ]
    point_on_plane = mricoords[first_non_centre_ind]

    for normal_vector in range(0, len(normal_components)):
        x = normal_components[normal_vector][0]
        y = normal_components[normal_vector][1]
        z = normal_components[normal_vector][2]

        if np.sum(np.array([x, y, z])) < 0:
            normal_components[normal_vector] = (
                normal_components[normal_vector] * -1
            )

    height = 4
    radius = 12
    empty_img = np.zeros(nifti.shape)
    cylinder_masks = [cylinder(nifti, c, radius, height) for c in centres]

    rotation_matrices = [
        get_rotation_matrix(np.array([0, 0, 1]), n) for n in normal_components
    ]

    rotated_cylinder_inds = np.vstack(
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

    rotated_cylinder = fill_holes(rotated_cylinder)

    save_nifti(img=rotated_cylinder, path=cylinder_mask_path, ref=nifti)

    # add plug

    plug_height = height + 10
    plug_radius = radius / 2

    plug_masks = [
        cylinder(nifti, c, plug_radius, plug_height) for c in centres
    ]

    rotated_plugs_inds = np.vstack(
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

    save_nifti(
        img=rotated_cyl_plus_plugs, path=cylinder_mask_plus_plug, ref=nifti
    )
