import logging
import os.path as op
import re

import SimpleITK as sitk
from alive_progress import alive_it
from paths import root_dir
from paths_funcs import glob_sub_dir


def mask_image(image, mask):
    """
    Mask image with mask

    Parameters
    ----------
    image:
        SimpleITK image
    mask:
        SimpleITK image

    Returns
    -------
    masked image
    """
    # Apply mask to image
    masker = sitk.MaskImageFilter()
    masker.SetMaskingValue(1)
    masker.SetOutsideValue(0)
    masked_image = masker.Execute(image, mask)

    subtracter = sitk.SubtractImageFilter()
    masked_image = subtracter.Execute(image, masked_image)

    # Return masked image
    return masked_image


def huang_threshold(image):
    """
    Huang thresholding

    Parameters
    ----------
    image:
        SimpleITK image

    Returns
    -------
    thresholded image
    """
    # Apply Huang thresholding
    huang_filter = sitk.HuangThresholdImageFilter()
    huang_filter.SetInsideValue(0)
    huang_filter.SetOutsideValue(1)
    thresholded_image = huang_filter.Execute(image)

    # Return thresholded image
    return thresholded_image


if __name__ == "__main__":
    sub_dirs = glob_sub_dir(root_dir)

    if sub_dirs is None:
        raise FileNotFoundError("No sub-directories found.")

    for sub_dir in alive_it(sub_dirs):
        logging.basicConfig(level=logging.INFO)
        cylinder_mask_path = op.join(sub_dir, "cylinder_ROI.nii.gz")
        # cylinder_mask_path = op.join(sub_dir, "cylinder_plus_plug_ROI.nii.gz")
        masked_petra = op.join(sub_dir, "petra_masked.nii.gz")
        refined_mask = op.join(sub_dir, "refined_electrode_ROI.nii.gz")

        sub, ses, run = re.findall(
            r"(sub-[0-9]+|ses-[0-9]+|run-[0-9]+)", sub_dir
        )

        if not op.exists(masked_petra):
            continue
        if not op.exists(cylinder_mask_path):
            continue
        if op.exists(refined_mask):
            continue

        logging.info(f"Refining cylinder ROI for {sub}, {ses}, {run}")

        petra = sitk.ReadImage(masked_petra, sitk.sitkFloat32)
        mask = sitk.ReadImage(cylinder_mask_path, sitk.sitkUInt8)

        try:
            masked_petra = mask_image(petra, mask)
        except RuntimeError:
            logging.error(f"Error in masking {masked_petra}")
            continue

        thresholded_petra = huang_threshold(masked_petra)

        sitk.WriteImage(thresholded_petra, refined_mask)
