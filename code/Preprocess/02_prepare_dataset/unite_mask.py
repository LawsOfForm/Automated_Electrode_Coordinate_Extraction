#use with mamba environment network
import nibabel as nib
import os.path as op
from glob import glob
import numpy as np
from tqdm.contrib import tzip
from pathlib import Path

SUBJECT_XNAT_PATH = op.join(
    "/media",
    "MeMoSLAP_Subjects",
    "derivatives",
    "automated_electrode_extraction",
    "sub-*",
    "unzipped",
)

def load_nifti(file: str) -> nib.Nifti1Image:
    return nib.load(file)

def save_nifti(file: str, nifti_img: nib.Nifti1Image) -> None:
    nib.save(nifti_img, file)

def main():
    files = glob(
        op.join(SUBJECT_XNAT_PATH, "sub-*_ses-*_run-0*", "Segmentation.nii")
    )

    dirs = [op.dirname(file) for file in files]

    print(f"Found {len(files)} files")

    nifti_imgs = [load_nifti(file) for file in files]
    canonical_imgs = [nib.as_closest_canonical(img) for img in nifti_imgs]
    
    imgs = [img.get_fdata() for img in canonical_imgs]
    affines = [img.affine for img in canonical_imgs]

    for i, img in enumerate(imgs):
        if img.shape != (224,288,288):
            print(f'problem with shape for {Path(dirs[i]).stem}:{img.shape}')

    # Filter out images with incorrect shape
    valid_indices = [i for i, img in enumerate(imgs) if img.shape == (224, 288, 288)]
    imgs = [imgs[i] for i in valid_indices]
    dirs = [dirs[i] for i in valid_indices]
    affines = [affines[i] for i in valid_indices]

    # Create binary masks
    imgs = [np.where(img > 0, 1, 0).astype(np.float32) for img in imgs]

    for dir, affine, img in tzip(dirs, affines, imgs):
        print(
            f"Saving {dir}\n"
            "--------------\n"
            f"unique values: {np.unique(img)}"
        )
        mask_img = nib.Nifti1Image(img, affine)
        save_nifti(op.join(dir, "mask.nii.gz"), mask_img)

if __name__ == "__main__":
    main()