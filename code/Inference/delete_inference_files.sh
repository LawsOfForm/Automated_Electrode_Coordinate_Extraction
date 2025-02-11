#!/bin/bash

base_path="/media/MeMoSLAP_DATA01/derivatives/automated_electrode_extraction"

# Loop through all subject directories
for subject_dir in "$base_path"/sub-*; do
    if [ -d "$subject_dir/unzipped" ]; then
        # Remove files with "inference" in the name and .nii.gz extension
        find "$subject_dir/unzipped" -type f -name "*inference*.nii.gz" -delete
    fi
done
