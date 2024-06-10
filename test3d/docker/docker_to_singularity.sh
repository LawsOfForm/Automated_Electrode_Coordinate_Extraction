#!/bin/bash
IMAGE_NAME="test/model:0.0"
SINGULARITY_NAME="model.sif"

docker build -t $IMAGE_NAME .
docker image save $IMAGE_NAME -o tmp.tar
sudo singularity build $SINGULARITY_NAME docker-archive://tmp.tar

rm tmp.tar
