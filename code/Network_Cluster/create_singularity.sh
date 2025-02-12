#!/bin/bash

# Set the output folder
OUTPUT_FOLDER="output"

# Create the output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Define the Singularity container name
CONTAINER_NAME="ubuntu-22.04-nvidia-py3.10.sif"

# Build the Singularity container
singularity build "$CONTAINER_NAME" docker://ubuntu:22.04

# Install NVIDIA runtime and Python 3.10 inside the container
singularity exec --writable "$CONTAINER_NAME" /bin/bash <<'EOF'
apt-get update
apt-get install -y software-properties-common
add-apt-repository universe
apt-get update
apt-get install -y nvidia-cuda-toolkit
apt-get install -y python3.10 python3.10-venv
EOF

# Run the module.py script inside the container and capture the output
OUTPUT=$(singularity exec "$CONTAINER_NAME" python3.10 module.py)

# Write the output to a file in the output folder
echo "$OUTPUT" > "$OUTPUT_FOLDER/output.txt"

echo "Output written to $OUTPUT_FOLDER/output.txt"
