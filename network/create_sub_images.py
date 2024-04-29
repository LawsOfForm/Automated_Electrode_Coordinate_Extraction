import logging
import os.path as op
import pickle
import re
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd
from alive_progress import alive_it
from get_project_data import get_project


def read_centre_surround(fname_pkl: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the MeMoSlap pkl file which stores the
    anode and cathode positions

    Parameter
    ----------
    fname_pkl(str): path to the pickle file

    Returns
    ----------
    tuple(np.ndarray, np.ndarray):
        anode and first cathode position
    """
    with open(fname_pkl, "rb") as pklfile:
        data = pickle.load(pklfile)

    centre_pos = data[1]
    if len(data[2]) > 1:
        raise ValueError(
            "Surround positions for more than one surround radius found.\n"
            "Not sure which radius to use"
        )

    keyname = list(data[2].keys())[0]
    surround_pos = data[2][keyname][0]

    return centre_pos, surround_pos


def ras2ijk(qform: np.ndarray, ras_coord: np.ndarray) -> np.ndarray:
    """
    Convert RAS coordinates to voxel coordinates using the qform matrix.

    Parameter
    ---------
    qform(4x4 matrix): transformation from individual space to world space
    ras_coord(1x3 matrix): coordinates in RAS space. Coordinate will be
        appended to make matrix multiplication possible.

    Returns
    ---------
    Converted Coordinates voxel coordinate system (1x4).
    """
    return np.linalg.solve(qform, np.hstack((ras_coord, 1)))


def get_sub_project(sub_id: str, assignment: pd.DataFrame) -> str:
    """
    Use the assignment table to get the subject's project assignment

    Parameter
    ----------
    sub-id(str): SubjectID
    assignment(pd.DataFrame): Dataframe that has the assignment stored

    Returns
    ---------
    str: the project the subject was assigned to
    """
    return assignment[assignment.SubjectID == sub_id].Project.iloc[0]


def create_crop(
    coords: np.ndarray,
    size: int | list[int] | np.ndarray,
    shape,
) -> np.ndarray:
    """
    Create the region where the image gets cropped. Respect logical
    limits like the shape of the image or zero.

    Parameters
    -----------
    coords(np.ndarray): center of the crop
    size(int, list[int]): Kernel of the crop.
        * int: Image gets cropped to a cube, i.e., int is used in
            all dimensions
        * list[int]: Image gets cropped to the dimensions specified
            in the list
    shape: Image shape

    Returns
    ---------
    3 x 2 np.ndarray with the upper and lower limits of x, y, and z
        dimension:

        [[lower_x, upper_x],
         [lower_y, upper_y],
         [lower_z, upper_z]]

    """
    if isinstance(size, int):
        size = [size, size, size]
    if isinstance(size, list):
        size = np.array(size, dtype=np.int16)
    shape = np.array(shape)

    lower = coords - size
    lower = np.where(lower < 0, 0, lower)

    upper = coords + size
    upper = np.where(upper > shape, shape, upper)

    return np.vstack((lower, upper)).transpose()


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG, filename="create_sub_images.log")
    CUBE_SIZE = 75

    petra_path = op.join(
        "/media",
        "MeMoSLAP_Subjects",
        "derivatives",
        "automated_electrode_extraction",
        "sub-*",
        "electrode_extraction",
        "ses-*",
        "run-*",
        "petra_cut_brain.nii.gz",
    )

    files = glob(petra_path)

    project_assignment = get_project()

    for file in alive_it(files):
        sub, ses, run = re.findall(
            pattern="(sub-[0-9]+|ses-[0-9]|run-[0-9]+)", string=file
        )

        if not any(project_assignment.SubjectID == sub):
            logging.warning("%s not found. Skipping", sub)
            continue

        project = get_sub_project(sub_id=sub, assignment=project_assignment)

        pkl_path = op.join(
            "/media",
            "MeMoSLAP_Mesh",
            "PDF_Report_Generation",
            "sham",
            "02-ANALYSIS",
            f"{project}_target_{sub[-3:]}",
            "simnibs_memoslap_results.pkl",
        )

        if not op.exists(pkl_path):
            logging.warning("%s, %s pickle not found", sub, project)
            continue

        centre, _ = read_centre_surround(pkl_path)

        petra = nib.load(file)  # pyright: ignore [reportPrivateImportUsage]
        voxel_coords = ras2ijk(
            qform=petra.get_qform(),  # pyright: ignore [reportAttributeAccessIssue]
            ras_coord=centre,
        )
        voxel_coords = voxel_coords.astype(np.int16)

        petra_img = np.array(
            petra.dataobj  # pyright: ignore [reportAttributeAccessIssue]
        )

        cropped_img = petra_img.copy()

        crop_margin = create_crop(
            coords=voxel_coords[:3],
            size=CUBE_SIZE,
            shape=cropped_img.shape,
        )

        cropped_img[: crop_margin[0, 0]] = 0
        cropped_img[crop_margin[0, 1] :] = 0
        cropped_img[:, : crop_margin[1, 0], :] = 0
        cropped_img[:, crop_margin[1, 1] :, :] = 0
        cropped_img[:, :, : crop_margin[2, 0]] = 0
        cropped_img[:, :, crop_margin[2, 1] :] = 0

        cropped_nifti = nib.Nifti1Image(  # pyright: ignore [reportPrivateImportUsage]
            cropped_img,
            header=petra.header,
            affine=petra.affine,  # pyright: ignore [reportAttributeAccessIssue]
        )

        base_out = op.dirname(file)

        nib.save(  # pyright: ignore [reportPrivateImportUsage]
            img=cropped_nifti,
            filename=op.join(
                base_out,
                # f"{sub}_{ses}_acq-petra_{run}_desc-region-around-electrode_PDw.nii.gz",
                "petra_desc-region_around_electrode.nii.gz",
            ),
        )
