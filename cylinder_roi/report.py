import os
import os.path as op
import re
from glob import glob

import matplotlib.pyplot as plt
from alive_progress import alive_it
from paths import root_dir
from paths_funcs import glob_sub_dir
from scipy.ndimage import center_of_mass
from util.io import load_nifti

sub_dirs = glob_sub_dir(root_dir)

if sub_dirs is None:
    raise FileNotFoundError("No sub-directories found.")

report_dir = op.dirname(op.abspath(__file__))
report_dir = op.join(report_dir, "reports")

if not op.exists(report_dir):
    os.mkdir(report_dir)

for sub_dir in alive_it(sub_dirs):
    sub, ses, run = re.findall(r"(sub-[0-9]+|ses-[0-9]+|run-[0-9]+)", sub_dir)

    report = op.join(report_dir, f"{sub}_{ses}_{run}.png")

    if op.isfile(report):
        continue

    masks = glob(op.join(sub_dir, "mask_*.nii.gz"))

    if not masks:
        continue

    petra_nii, petra = load_nifti(op.join(sub_dir, "petra_.nii.gz"))
    petra_max = petra.max()

    _, ax = plt.subplots(len(masks), 3, figsize=(12, 12))
    for i, mask in enumerate(masks):
        mask_nii, mask = load_nifti(mask)

        com = center_of_mass(mask)

        mask_overlayed_petra = petra.copy()
        mask_overlayed_petra[mask == 1] = petra_max

        slice_0 = mask_overlayed_petra[int(com[0]), :, :]
        slice_1 = mask_overlayed_petra[:, int(com[1]), :]
        slice_2 = mask_overlayed_petra[:, :, int(com[2])]

        slices = [slice_0, slice_1, slice_2]

        for j, slice_ in enumerate(slices):
            ax[i][j].imshow(slice_, cmap="gray")
            plt.axis("off")

    plt.savefig(report)
    plt.close()
