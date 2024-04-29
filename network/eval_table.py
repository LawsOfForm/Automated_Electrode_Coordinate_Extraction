import os.path as op
from glob import glob

import pandas as pd

base_path = op.join(
    "/media",
    "data01",
    "sriemann",
    "MeMoSlap",
    "cubes",
)
images = glob(op.join(base_path, "sub-*_ses-*_run-*_volume-*.nii.gz"))
images.sort()

eval_table = pd.DataFrame({"path": images})

eval_table = eval_table.assign(
    sub="sub-" + eval_table.path.str.extract(r"sub-([0-9]+)"),
    ses="ses-" + eval_table.path.str.extract(r"ses-([0-9]+)"),
    run="run-" + eval_table.path.str.extract(r"run-([0-9]+)"),
    vol="volume-" + eval_table.path.str.extract(r"volume-([0-9]+)"),
    good_segmentation=True,
    n_electrodes=0,
)

eval_table.to_csv("eval_table.tsv", sep="\t", index=False)
