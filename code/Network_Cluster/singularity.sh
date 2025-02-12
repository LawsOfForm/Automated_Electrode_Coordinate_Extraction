#!/bin/bash

path="/home/niemannf/Documents/code_for_brain_cluster"

# Define the container name and directory
CONTAINER_NAME="monai-pytorch-cuda1700"
CONTAINER_DIR="$path/singularity/$CONTAINER_NAME"

# Create the container directory if it doesn't exist
mkdir -p "$CONTAINER_DIR"

# Define the Singularity definition file
DEFINITION_FILE="$CONTAINER_DIR/Singularity_cuda1700.def"

# Create the Singularity definition file
cat << EOF > "$DEFINITION_FILE"
BootStrap: docker
From: ubuntu:22.04
#From: nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu22.04
From: nvidia/cuda:11.7-cudnn8-runtime-ubuntu22.04
From: python:3.10

%post
    # Update package lists and install dependencies
    apt-get update
    apt-get install -y software-properties-common
    apt-get install -y wget curl git build-essential

    # Install CUDA toolkit
    wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
    sh cuda_11.7.1_515.65.01_linux.run --silent --toolkit --override

    # Install Python 3.10
    #apt-get install -y python3.10 python3.10-venv python3.10-dev
    apt-get install -y python3.10

    # Create a Python virtual environment
    #python3.10 -m venv /opt/monai-env
    #. /opt/monai-env/bin/activate
  
    
    # Install PyTorch with CUDA support
    pip install torch==2.1.1 --extra-index-url https://download.pytorch.org/whl/cu117
    #pip install torch==2.3.1 

    # Install MONAI
    pip install monai[all]==1.3.1
    
    # Install matplotlib
    pip install matplotlib==3.9.0

    # Install matplotlib
    pip install numpy==1.26.2
    
    
    # Clean up
    apt-get clean
    rm -rf /var/lib/apt/lists/*

%environment
    #export PATH="/opt/monai-env/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

%runscript
    #echo "Starting MONAI and PyTorch environment..."
    #. /opt/monai-env/bin/activate
EOF

# Build the Singularity container
sudo singularity build "$CONTAINER_DIR/$CONTAINER_NAME.sif" "$DEFINITION_FILE"

echo "Container built successfully: $CONTAINER_DIR/$CONTAINER_NAME.sif"
