"""
Configuration file with constants for `docker_model.py`

CONSTANTS:
    SUBJECT_PATTERN: str
        Pattern for subject directories. Just add an asterisk (*)
        for every changing part of the path.
    VOLUME_SUFFIX: str
        How the volume files are named. Could also include a path
        or asterisks to indicate a pattern, but should only
        lead to one file for every subject, session, run combination.
    MASK_SUFFIX: str
        How the mask files are named. Could also include a path
        or asterisks to indicate a pattern, but should only
        lead to one file for every subject, session, run combination.
    INPUT_DIR: str
        Path to the input directory. This is how the virtual volume
        should be named in the Docker/Singularity container.
    OUTPUT_DIR: str
        Path to the output directory. This is how the virtual output
        directory should be named in the Docker/Singularity container.
"""


SUBJECT_PATTERN = "sub-*/electrode_extraction/ses-*/run-*"
VOLUME_SUFFIX = "petra_.nii.gz"
MASK_SUFFIX = "petra_masked.nii.gz"
INPUT_DIR = "/data"
OUTPUT_DIR = "/results"
