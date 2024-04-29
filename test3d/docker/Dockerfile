# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

################################################################################
# Pick a base image to serve as the foundation for the other build stages in
# this file.
#
# For illustrative purposes, the following FROM command
# is using the alpine image (see https://hub.docker.com/_/alpine).
# By specifying the "latest" tag, it will also use whatever happens to be the
# most recent version of that image when you build your Dockerfile.
# If reproducibility is important, consider using a versioned tag
# (e.g., alpine:3.17.2) or SHA (e.g., alpine@sha256:c41ab5c992deb4fe7e5da09f67a8804a46bd0592bfdf0b1847dde0e0889d2bff).
FROM ubuntu:22.04 as base

RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
  && localedef -i de_DE -c -f UTF-8 -A /usr/share/locale/locale.alias de_DE.UTF-8
ENV LANG de_DE.utf8

FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

################################################################################
# Create a stage for building/compiling the application.
#
# The following commands will leverage the "base" stage above to generate
# a "hello world" script and make it executable, but for a real application, you
# would issue a RUN command for your application's build process to generate the
# executable. For language-specific examples, take a look at the Dockerfiles in
# the Awesome Compose repository: https://github.com/docker/awesome-compose

FROM python:3.10

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY ./model.py .

# ENTRYPOINT ["/bin/bash"]
# ENTRYPOINT ["python", "model.py"] # add arguments to the call
ENTRYPOINT ["python", "-c", "import torch; print(torch.cuda.is_available())"]
