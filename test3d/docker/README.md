# Build a docker container to singularity script

There are multiple methods documented online. On Linux-Meinzer2
are some dependencies problems that do not allow us to use some.
However, the method below works:

1. Build a docker image, where `.` points to the path of the Dockerfile

```bash
#docker build -t <user>/<image_name>:version .
docker build -t test/model:0.1 .
```

2. Save the docker image as `tar`

```bash
docker image save test/model:0.1 test_model.tar
```

3. Build the singularity container from the docker archive

```bash
sudo singularity build test_model.tar docker-archive://test_model.tar
```

# Run the Docker skript

```bash
docker run --rm -it -v /path/to/input/data:/data:ro -v /output/path/to/store/results/:/results test/model:0.1
```

# Run the Singularity skript

```bash
singularity run -B /path/to/input/data:/data:ro -B /output/path/to/store/results/:/results test_model.sif
```
