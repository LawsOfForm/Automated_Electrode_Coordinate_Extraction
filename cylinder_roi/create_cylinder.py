import logging
import os.path as op
import re
from glob import glob

import cv2 as cv
import nibabel as nib
import numpy as np
import scipy
from alive_progress import alive_it
from packaging import version
from scipy.spatial.transform import Rotation as R


def circle_mask(xdim, ydim, radius, centre):
    Y, X = np.ogrid[:ydim, :xdim]
    dist_from_centre = np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2)

    return dist_from_centre <= radius


def cylinder(img, centre, radius, height):
    cylinder_img = np.zeros(img.shape)

    for z in range(centre[2], centre[2] + height):
        mask = circle_mask(
            img.shape[0],
            img.shape[1],
            radius,
            centre[:2],
        )
        cylinder_img[:, :, z] = mask.transpose()

    return cylinder_img


def get_rotation_matrix(a, b):
    u_normal = a / np.linalg.norm(a)
    v_normal = b / np.linalg.norm(b)

    v = np.cross(u_normal, v_normal)

    v_skew_symmetric = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
    )

    rotation_matrix = (
        np.eye(3)
        + v_skew_symmetric
        + np.dot(v_skew_symmetric, v_skew_symmetric)
        * 1
        / (1 + np.dot(u_normal, v_normal))
    )

    return rotation_matrix


def create_rotation_matrix(v1, v2):
    # Normalisieren Sie die Vektoren
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Berechnen Sie die Rotationsachse und den Winkel
    axis = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2))

    # Erstellen Sie die Rotationsmatrix
    rotation = R.from_rotvec(axis * angle)
    if version.parse(scipy.__version__) >= version.parse("1.4.0"):
        return rotation.as_matrix()
    return rotation.as_dcm()


def pad_matrix(matrix):
    matrix_pad = np.eye(4)
    matrix_pad[:3, :3] = matrix
    return matrix_pad


def rotate_img_obj(img, rotation_matrix, centre):
    centre = centre.reshape(3, 1)
    cylinder_ind = np.where(img == 1)
    centered_cylinder_ind = cylinder_ind - centre
    rotated_centered_cylinder_ind = rotation_matrix @ centered_cylinder_ind
    rotated_cylinder_ind = (rotated_centered_cylinder_ind + centre).T
    rotated_cylinder_ind = np.vstack(
        (
            np.floor(rotated_cylinder_ind),
            np.ceil(rotated_cylinder_ind),
        )
    ).astype("int32")

    return np.unique(rotated_cylinder_ind, axis=0)


def fill_holes(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    return img


def read_mricoords(path):
    return scipy.io.loadmat(path)["mricoords"].T


def get_normal_component(mricoords):
    return np.cross(
        mricoords[1] - mricoords[0],
        mricoords[2] - mricoords[0],
    )


def project_onto_plane(
    point: np.ndarray,
    plane_normal: np.ndarray,
    plane_point: np.ndarray,
) -> np.ndarray:
    """
    Project a point onto a plane.

    Parameters
    ----------
    point : np.ndarray
        The point to project.
    plane_normal : np.ndarray
        The normal vector of the plane.
    plane_point : np.ndarray
        A point on the plane.

    Returns
    -------
    np.ndarray
        The projected point.
    """

    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    dist_to_plane = np.dot(point - plane_point, plane_normal)

    return point - dist_to_plane * plane_normal


if __name__ == "__main__":
    root_dir = op.join(
        "/media",
        "MeMoSLAP_Subjects",
        "derivatives",
        "automated_electrode_extraction",
    )
    sub_dirs = glob(
        op.join(
            root_dir,
            "sub-*",
            "electrode_extraction",
            "ses-*",
            "run-*",
        )
    )
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

        if mricoords.shape[0] != 12:
            logging.warning(
                f"sub-{sub} has {mricoords.shape[0]} electrode coordinates "
                + f"in ses-{ses}, run-{run}. Expected 12.\n"
                "Will skip subject."
            )
            continue
        logging.info(
            f"Creating cylinder ROI for sub-{sub}, ses-{ses}, run-{run}"
        )

        centres_ind = np.arange(0, 12, 3)
        centres = mricoords[centres_ind]
        normal_components = [
            get_normal_component(mricoords[i : i + 3]) for i in centres_ind
        ]

        height = 5
        radius = 10
        empty_img = np.zeros(nifti.shape)

        cylinder_masks = [cylinder(nifti, c, radius, height) for c in centres]

        rotation_matrices = [
            get_rotation_matrix(np.array([0, 0, 1]), n)
            for n in normal_components
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
