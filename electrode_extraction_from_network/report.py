import os
import os.path as op
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_it
from paths import root_dir
from paths_funcs import glob_sub_dir
from scipy.ndimage import center_of_mass
from util.io import load_nifti


def slice_at(img, coords):
    return [
        img[int(coords[0]), :, :],
        img[:, int(coords[1]), :],
        img[:, :, int(coords[2])],
    ]


def normalize_img(img, clip=True, rng=(10, 99)):
    if clip:
        p95 = np.percentile(img, rng[1])
        p10 = np.percentile(img, rng[0])
        img = np.clip(img, p10, p95)

    img = ((img - img.min()) / (img.max() - img.min())) * 255
    return img.astype(np.uint8)


sub_dirs = glob_sub_dir(root_dir)

if sub_dirs is None:
    raise FileNotFoundError("No sub-directories found.")

report_dir = op.dirname(op.abspath(__file__))
report_dir = op.join(report_dir, "reports_inference")

if not op.exists(report_dir):
    os.mkdir(report_dir)

for sub_dir in alive_it(sub_dirs):
    sub, ses, run = re.findall(r"(sub-[0-9]+|ses-[0-9]+|run-[0-9]+)", sub_dir)

    report = op.join(report_dir, f"{sub}_{ses}_{run}.png")

    # if op.isfile(report):
    #     continue
    #
    #masks = glob(op.join(sub_dir, "mask_*.nii.gz"))
    masks = glob(op.join(sub_dir, "mask_*_inference.nii.gz"))
    if not masks:
        continue

    petra_nii, petra = load_nifti(op.join(sub_dir, "petra_.nii.gz"))
    normalized_petra = normalize_img(petra, rng=(10, 99))

    _, axs = plt.subplots(len(masks), 3, figsize=(12, 12))
    for i, mask in enumerate(masks):
        mask_nii, mask = load_nifti(mask)

        com = center_of_mass(mask)

        petra_slices = slice_at(normalized_petra, com)

        mask_slices = slice_at(mask, com)

        for j, (petra_slice, mask_slice) in enumerate(
            zip(petra_slices, mask_slices)
        ):
            axs[i][j].imshow(petra_slice, cmap="gray")
            axs[i][j].imshow(mask_slice.squeeze(), cmap="Reds", alpha=0.3)
            plt.axis("off")

    plt.savefig(report)
    plt.close()
