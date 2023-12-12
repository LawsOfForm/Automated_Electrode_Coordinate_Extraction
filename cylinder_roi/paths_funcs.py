import os.path as op
from glob import glob


def sub_directory(root_dir: str, sub: int, ses: int, run: int) -> str:
    return op.join(
        root_dir,
        f"sub-{sub:03d}",
        "electrode_extraction",
        f"ses-{ses}",
        f"run-{run:02d}",
    )


def glob_sub_dir(root_dir: str) -> list[str] | None:
    sub_dirs = glob(
        op.join(
            root_dir,
            "sub-*",
            "electrode_extraction",
            "ses-*",
            "run-*",
        )
    )

    if sub_dirs:
        return sub_dirs
