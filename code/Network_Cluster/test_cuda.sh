#!/bin/bash

# Set the name of the Singularity image
IMAGE_NAME="mod.sif"

# Set the path to the Python script
SCRIPT_PATH="test_cuda.py"

# Create a temporary directory for the output
OUTPUT_DIR=$(mktemp -d)

# Run the Singularity container with the Python script
singularity exec --nv \
    --bind "$OUTPUT_DIR":/output \
    "$IMAGE_NAME" \
    python "$SCRIPT_PATH" > "/output/output.txt"

# Print the output
cat "$OUTPUT_DIR/output.txt"

# Clean up the temporary directory
rm -rf "$OUTPUT_DIR"
