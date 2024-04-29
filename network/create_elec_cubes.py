import logging
import os
import os.path as op
import re
from glob import glob

import nibabel as nib
import numpy as np
from alive_progress import alive_it
from create_sub_images import (
    create_crop,
    get_sub_project,
    ras2ijk,
    read_centre_surround_all,
)
from get_project_data import get_project


def cut_cubes(img: np.ndarray, cut_coords: np.ndarray):

    return [
        img[
            coords[0, 0] : coords[0, 1],
            coords[1, 0] : coords[1, 1],
            coords[2, 0] : coords[2, 1],
        ]
        for coords in cut_coords
    ]


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG, filename="create_sub_images.log")
    CUBE_SIZE = 32

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

        elec_coords = read_centre_surround_all(pkl_path)

        petra = nib.load(file)  # pyright: ignore [reportPrivateImportUsage]
        voxel_coords = np.vstack(
            [
                ras2ijk(
                    qform=petra.get_qform(),  # pyright: ignore [reportAttributeAccessIssue]
                    ras_coord=coord,
                )[:3]
                for coord in elec_coords
            ]
        )

        voxel_coords = voxel_coords.astype(np.int16)

        petra_img = np.array(
            petra.dataobj  # pyright: ignore [reportAttributeAccessIssue]
        )

        N_ELECTRODES = 4
        N_DIM = 3
        N_THRES = 2
        crop_margin = np.vstack(
            [
                create_crop(
                    coords=coord,
                    size=[20, CUBE_SIZE, CUBE_SIZE],
                    shape=petra_img.shape,
                )
                for coord in voxel_coords
            ]
        ).reshape(N_ELECTRODES, N_DIM, N_THRES)

        cropped_imgs = cut_cubes(img=petra_img, cut_coords=crop_margin)

        mask = nib.load(  # pyright: ignore [reportPrivateImportUsage]
            filename=op.join(op.dirname(file), "cylinder_plus_plug_ROI.nii.gz"),
        )
        mask_img = np.array(
            mask.dataobj  # pyright: ignore [reportAttributeAccessIssue]
        )  # pyright: ignore [reportAttributeAccessIssue]

        cropped_mask = cut_cubes(img=mask_img, cut_coords=crop_margin)

        # cropped_nifti = nib.Nifti1Image(  # pyright: ignore [reportPrivateImportUsage]
        #     cropped_img,
        #     header=petra.header,
        #     affine=petra.affine,  # pyright: ignore [reportAttributeAccessIssue]
        # )

        base_out = op.join(op.dirname(file), "cut_cubes")

        if not op.exists(base_out):
            os.mkdir(base_out)

        for idx, (img, mask) in enumerate(zip(cropped_imgs, cropped_mask)):

            print(f"mask shape:\n{mask.shape}")
            print(f"vol shape:\n{img.shape}")
            nib.save(  # pyright: ignore [reportPrivateImportUsage]
                img=nib.Nifti1Image(mask, header=petra.header, affine=petra.affine),
                filename=op.join(base_out, f"mask_{idx + 1}.nii.gz"),
            )
            nib.save(  # pyright: ignore [reportPrivateImportUsage]
                img=nib.Nifti1Image(img, header=petra.header, affine=petra.affine),
                filename=op.join(base_out, f"volume_{idx + 1}.nii.gz"),
            )

        # nib.save(  # pyright: ignore [reportPrivateImportUsage]
        #     img=cropped_nifti,
        #     filename=op.join(
        #         base_out,
        #         # f"{sub}_{ses}_acq-petra_{run}_desc-region-around-electrode_PDw.nii.gz",
        #         "petra_desc-region_around_electrode.nii.gz",
        #     ),
        # )
