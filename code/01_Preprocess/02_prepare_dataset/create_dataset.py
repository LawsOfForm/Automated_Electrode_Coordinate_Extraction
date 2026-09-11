#use with mamba environment network
from glob import glob
import os.path as op
import os
from unite_mask import SUBJECT_XNAT_PATH
import re
from joblib import Parallel, delayed
import tarfile
import pathlib



def info_from_path(path: str):
    match = re.match(
        r".*sub-(?P<subject>.*)_ses-(?P<session>.*)_run-(?P<run>.*)/",
        path,
    )

    if match is None:
        raise ValueError(f"Could not match {path}")

    return {
        "subject": match.group("subject"),
        "session": match.group("session"),
        "run": match.group("run"),
    }


def vol_path_from_info(info: dict, path: str):
    vol_path = path.replace("sub-*", f"sub-{info['subject']}", 2)
    vol_path = vol_path.replace("ses-*", f"ses-{info['session']}", 1)
    vol_path = vol_path.replace("run-*", f"run-{info['run']}", 1)
    return vol_path


def main():
    masks = glob(op.join(SUBJECT_XNAT_PATH, "sub-*_ses-*_run-0*/mask.nii.gz"))

    infos = Parallel(n_jobs=-1)(
        delayed(info_from_path)(mask) for mask in masks
    )

    vols = Parallel(n_jobs=-1)(
        delayed(vol_path_from_info)(
            info,
            op.join(SUBJECT_XNAT_PATH, "rsub-*_ses-*_acq-petra_run-*_PDw.nii"),
        )
        for info in infos
    )

    path_dataset=pathlib.Path(__file__).parent.resolve()

    with tarfile.open(os.path.join(path_dataset,"dataset.tar.gz"), "w:gz") as tar:
        for mask, vol in zip(masks, vols):
            tar.add(mask)
            tar.add(vol)


if __name__ == "__main__":
    main()
