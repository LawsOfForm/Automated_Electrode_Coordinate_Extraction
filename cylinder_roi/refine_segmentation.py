import os.path as op

import nibabel as nib
import numpy as np
from paths import glob_sub_dir, root_dir
from skimage.feature import canny
from skimage.morphology import convex_hull_image


def load_nifti(
    path: str,
) -> tuple[nib.filebasedimages.FileBasedImage, np.ndarray]:
    nifti = nib.load(path)
    img = nifti.dataobj
    return (nifti, np.array(img))


def save_nifti(
    img: np.ndarray, path: str, ref: nib.filebasedimages.FileBasedImage
) -> None:
    new_img = nib.nifti1.Nifti1Image(
        img,
        ref.affine,
        ref.header,
    )
    nib.save(new_img, path)


if __name__ == "__main__":
    sub = 10
    run = 1
    ses = 1

    sub_dirs = glob_sub_dir(root_dir)

    for sub_dir in sub_dirs:
        petra_path = op.join(sub_dir, "petra_.nii.gz")
        cylinder_mask_path = op.join(sub_dir, "cylinder_test.nii.gz")

        petra, petra_img = load_nifti(petra_path)
        _, mask_img = load_nifti(cylinder_mask_path)

        masked_petra = petra_img * mask_img

        sigma = 1

        edges = np.array([canny(img, sigma=sigma) for img in masked_petra])
        chull = np.array(
            [
                convex_hull_image(img) if len(np.unique(img)) > 1 else img
                for img in edges
            ]
        )

        save_nifti(
            edges, op.join(sub_dir, f"canny_sigma_{sigma}.nii.gz"), petra
        )
        save_nifti(chull, op.join(sub_dir, "chull.nii.gz"), petra)

    # hough = np.array(
    #     [
    #         hough_ellipse(
    #             img,
    #         )
    #         for img in edges
    #     ]
    # )
