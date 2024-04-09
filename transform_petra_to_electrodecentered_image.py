# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:44:13 2024

@author: axthi
"""

import os
import pickle
from functools import partial

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes
from scipy.spatial.transform import Rotation as rot
from simnibs import file_finder, mesh_io
from simnibs.segmentation.samseg import gems
from simnibs.utils.transformations import volumetric_affine


def reg_6DOF(fixed_scan, moving_scan, output):
    RAS2LPS = np.diag([-1, -1, 1, 1])
    reg = gems.KvlRigidRegistration()
    reg.read_images(fixed_scan, moving_scan)
    reg.initialize_transform()
    reg.register()
    trans_mat = RAS2LPS @ reg.get_transformation_matrix() @ RAS2LPS
    reg.write_out_result(output)

    path_name = os.path.split(output)
    filename = path_name[1]
    filename = filename.split(".")
    filename = filename[0]
    mat_path = os.path.join(path_name[0], filename + "_dof6.dat")
    np.savetxt(mat_path, trans_mat)
    return trans_mat


def jitter_matrix(trans_bounds, rot_bounds, scale_bounds):
    """returns an affine matrix with random scaling, rotation
    and translation (in this order).
    Scaling is relative (1.0 is no scaling)
    Rotation is in degrees; order of rotation: z, x, y (extrinsic axes)
    Translation is in millimeter
    """
    t_vec = [
        np.random.uniform(trans_bounds["x"][0], trans_bounds["x"][1]),
        np.random.uniform(trans_bounds["y"][0], trans_bounds["y"][1]),
        np.random.uniform(trans_bounds["z"][0], trans_bounds["z"][1]),
    ]
    T = np.eye(4)
    T[:3, 3] = t_vec

    r_angles = [
        np.random.uniform(rot_bounds["x"][0], rot_bounds["x"][1]),
        np.random.uniform(rot_bounds["y"][0], rot_bounds["y"][1]),
        np.random.uniform(rot_bounds["z"][0], rot_bounds["z"][1]),
    ]

    r_mat = rot.from_euler("xyz", r_angles, degrees=True)
    R = np.eye(4)
    R[:3, :3] = r_mat.as_matrix()

    s_fac = [
        np.random.uniform(scale_bounds["x"][0], scale_bounds["x"][1]),
        np.random.uniform(scale_bounds["y"][0], scale_bounds["y"][1]),
        np.random.uniform(scale_bounds["z"][0], scale_bounds["z"][1]),
        1,
    ]
    S = np.diag(s_fac)

    return T @ R @ S


def check_img_shapes(img1: nib.nifti1.Nifti1Image, img2: nib.nifti1.Nifti1Image):
    """Takes two nifti-images and checks if the first three dims are the same shape

    Args:
        img1 (nib.nifti1.Nifti1Image)
        img2 (nib.nifti1.Nifti1Image)

    Raises:
        ValueError: Nifit images have different dimensions
    """
    shape1 = img1.dataobj.shape[:3]
    shape2 = img2.dataobj.shape[:3]
    if not np.all(shape1 == shape2):
        raise ValueError(
            "Image1 shape != Image2 shape:\n"
            f"{shape1} != {shape2}\n"
            "Are they coregistered?"
        )


def check_affine(img1_aff, img2_aff):
    if not np.all(np.isclose(img1_aff, img2_aff)):
        raise ValueError(
            "Reference and electrode affines do not match\n" "Are they coregistered?"
        )


def load_electrode_img(
    ref_img: nib.nifti1.Nifti1Image,
    electrode_image: str,
    manlabel_image: str,
    register_to_T1: bool = False,
) -> nib.nifti1.Image:
    """Load electrode images (PETRA) and corresponding manual labels. Coregister
    them to the T1, which was the template for the SimNIBS simulations, if
    needed

    Args:
        ref_img (nib.nifti1.Nifti1Image): Reference image (T1) of SimNIBS simulations
        electrode_image (str): PETRA
        manlabel_image (str): Manually labeled electrodes
        register_to_T1 (bool, optional): Whether electrode images should be registered to the Reference image.
            Throws ValueError if they False and they weren't registered before. Defaults to False.

    Returns:
        nib.nifti1.Image: Electrode image (PETRA) coregistered to reference
    """

    ref_affine = ref_img.get_qform()
    ele_img = nib.load(electrode_image)
    manlabel_img = nib.load(manlabel_image)

    if not register_to_T1:
        ele_affine = ele_img.get_qform()
        # assert np.all(ref_img.dataobj.shape[:3] == ele_img.dataobj.shape[:3])
        # assert np.all(np.isclose(ele_affine, ref_affine))
        check_img_shapes(ref_img, ele_img)
        check_affine(ref_affine, ele_affine)

        manlabel_affine = manlabel_img.get_qform()
        manlabel_vol = manlabel_img.get_fdata().astype(np.int16)
        # assert np.all(ref_img.dataobj.shape[:3] == manlabel_img.dataobj.shape[:3])
        # assert np.all(np.isclose(manlabel_affine, ref_affine))
        check_img_shapes(ref_img, manlabel_img)
        check_affine(ref_img_affine, manlabel_affine)
    else:
        # register PETRA to T1
        reg_mat = reg_6DOF(
            sub_files.reference_volume,
            electrode_image,
            os.path.join(output_folder, "elec_img_coreg.nii.gz"),
        )

        # apply registration to manual label image
        hlp_img = nib.load(manlabel_image)
        hlp_affine = hlp_img.get_qform()
        hlp_vol = hlp_img.get_fdata().astype(np.int16)

        manlabel_vol = volumetric_affine(
            image=(hlp_vol, hlp_affine),
            affine=reg_mat,
            target_space_affine=ref_affine,
            target_dimensions=ref_img.dataobj.shape[:3],
            intorder=0,  # *NOTE: nearest-neighbor = 0, linear = 1, ... (quadratic = 2 ??)
        )
        hlp_img = nib.Nifti1Image(manlabel_vol, ref_affine)
        nib.save(hlp_img, os.path.join(output_folder, "manlabel_img_coreg.nii.gz"))

    return ele_img.get_fdata(), manlabel_vol


def mask_from_simnibs(path: str):
    """Use SimNIBS segmentation do generate a head mask that excludes
    everything that is too far away from the scalp.

    Args:
        path (str): path to SimNIBS labels file

    Returns:
        np.ndarray: Image mask
    """
    headmask_img = nib.load(sub_files.labeling)
    headmask_vol = np.squeeze(headmask_img.get_fdata().astype(np.int16))
    # ? i.e. 517 == volumes too far away from the skin surface
    headmask_vol[headmask_vol == 517] = 0
    headmask_vol = binary_fill_holes(headmask_vol > 0)

    T1_voxsize = np.sqrt(np.sum(ref_affine[:3, :3] ** 2, axis=0))
    nsteps_ero = int(boundary_belowskin / np.mean(T1_voxsize))
    nsteps_dil = int(boundary_aboveskin / np.mean(T1_voxsize))

    mask = binary_dilation(headmask_vol, iterations=nsteps_dil)
    mask ^= binary_erosion(headmask_vol, iterations=nsteps_ero)

    return mask


def read_centre_surround(fname_pkl):
    with open(fname_pkl, "rb") as pklfile:
        data = pickle.load(pklfile)
        center_pos = data[1]
        if len(data[2]) > 1:
            raise ValueError(
                "surround positions for more than one surround radius found. Not sure which radius to use"
            )
        keyname = list(data[2].keys())[0]
        # data[2][keyname] # *NOTE: was not commented in original, but seemed to have no effect
        surround_pos = data[2][keyname][0]

    return center_pos, surround_pos


if __name__ == "__main__":
    # %% settings
    sub = "002"
    m2m_folder = os.path.join("/media/MeMoSLAP_Mesh", f"m2m_{sub}")
    project_folder = "tests\\P6_target_ernie"  # path to pickles
    electrode_image = (
        "org\\ernie_T2.nii.gz"  # name of the PETRA image with the electrodes
    )
    manlabel_image = (
        "org\\ernie_T2.nii.gz"  # this is the manual label image (binary mask)
    )
    output_folder = "cylinder_test"

    register_to_T1 = (
        True  # set to True if input image is not yet coregistered with T1 of m2m-folder
    )
    write_resampledT1_as_control = True  # for debugging

    # set the size of the resampled image, and the z-offset of the origin
    # * Notes:
    #   * origin will be in the center of the image in x and y directions
    #   * resolution of the resampled image will be 1 mm isotropic
    #   * image shape has to be set large enough to reliably cover electrodes
    #          also for largest montage --> needs some piloting
    resample_shape = [200, 200, 80]
    resample_z_offset = 60
    # set how much to keep above and below skin boundary (in millimeter)
    boundary_aboveskin = 10
    boundary_belowskin = 20

    jitter_image = (
        True  # whether to apply a random affine matrix to create more training data
    )
    # max translations in millimeter
    trans_bounds = {"x": [-10, 10], "y": [-10, 10], "z": [-1, 5]}
    # max rotation in degrees
    rot_bounds = {"x": [-5, 5], "y": [-5, 5], "z": [0, 90]}
    # relative scaling factors
    scale_bounds = {"x": [0.75, 1.25], "y": [0.75, 1.25], "z": [0.9, 1.10]}

    # %% load electrode image (check img sizes and affines) or register
    sub_files = file_finder.SubjectFiles(subpath=m2m_folder)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    ref_img = nib.load(sub_files.reference_volume)  # NOTE: reference_volume = T1.nii.gz
    ref_vol = ref_img.get_fdata()
    ref_affine = ref_img.get_qform()
    ele_vol, manlabel_vol = load_electrode_img(
        ref_img, electrode_image, manlabel_image, register_to_T1
    )

    # %% use simnibs segmentation to blank out regions too far way from the skin surface
    mask = mask_from_simnibs(path=sub_files.labeling)
    hlp_img = nib.Nifti1Image(mask.astype(np.int16), ref_affine)
    nib.save(hlp_img, os.path.join(output_folder, "mask.nii.gz"))

    ele_vol[~mask] = 0
    manlabel_vol[~mask] = 0

    # %% get positions of center and first surround electrode from pickle
    fname_pkl = os.path.join(project_folder, "simnibs_memoslap_results.pkl")
    center_pos, surround_pos = read_centre_surround(fname_pkl=fname_pkl)
    # use simnibs function to project center position on skin and span up local coordinate system
    m = mesh_io.read_msh(sub_files.fnamehead)
    world_to_world_org = m.calc_matsimnibs(center_pos, surround_pos, 0)
    # rotate around y so that z points away from head
    world_to_world_org[:3, 0] = -world_to_world_org[:3, 0]
    world_to_world_org[:3, 2] = -world_to_world_org[:3, 2]

    # create affine matrix for resampled image:
    # set origin in middle of image (x,y) and at resample_z_offset (z)
    resample_affine = np.eye(4)
    resample_affine[:3, 3] = -(np.array(resample_shape) + 1) / 2
    resample_affine[2, 3] = -resample_z_offset

    # resample image and save
    # Note: the following part would be iterated (with jitter_image=True) to create sufficient training data
    world_to_world = (
        world_to_world_org @ jitter_matrix(trans_bounds, rot_bounds, scale_bounds)
        if jitter_image
        else world_to_world_org
    )

    partial_volumetric_affine = partial(
        volumetric_affine,
        affine=world_to_world,
        target_space_affine=resample_affine,
        target_dimensions=resample_shape,
    )

    if write_resampledT1_as_control:
        T1_resampled = partial_volumetric_affine((ref_vol, ref_affine))
        img = nib.Nifti1Image(T1_resampled, resample_affine)
        nib.save(img, os.path.join(output_folder, "T1_resampled.nii.gz"))

    Ele_resampled = partial_volumetric_affine((ele_vol, ref_affine))
    img = nib.Nifti1Image(Ele_resampled, resample_affine)
    nib.save(img, os.path.join(output_folder, "Ele_resampled.nii.gz"))

    Manlabel_resampled = partial_volumetric_affine(
        (manlabel_vol, ref_affine), intorder=0
    )
    img = nib.Nifti1Image(Manlabel_resampled, resample_affine)
    nib.save(img, os.path.join(output_folder, "Manlabel_resampled.nii.gz"))

    # in case you want to get the automated electrode segmentation back to the
    # original image space, you can adapt this:
    back_to_T1space = volumetric_affine(
        (Manlabel_resampled, resample_affine),
        np.linalg.inv(world_to_world),
        ref_affine,
        ref_vol.shape,
        intorder=0,
    )
    img = nib.Nifti1Image(back_to_T1space, ref_affine)
    nib.save(img, os.path.join(output_folder, "back_to_T1space.nii.gz"))
