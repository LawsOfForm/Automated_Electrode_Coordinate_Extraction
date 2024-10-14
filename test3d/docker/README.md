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

## Run the Docker skript

```bash
docker run --rm -it \
    -v /path/to/input/data:/data:ro \
    -v /output/path/to/store/results/:/results \
    test/model:0.1
```

## Run the Singularity skript

```bash
singularity run --nv \ # --nv necessary for GPU support
    -B /path/to/input/data:/data }
    -B /output/path/to/store/results/:/results \
    test_model.sif
```

## Run the script on the HPC Greifswald

Copy the singularity container to the HPC.

```bash
scp test_model.sif <user>@brain.uni-greifswald.de:~/path/to/store/
```

Write a submit file. Insert your email-address.

```bash
#!/bin/bash
#SBATCH -J unet_train
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=vision
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -t 72:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=<email-address>

module load cuda/12.0.0
module load singularity/3.11.3

singularity run --nv \
    -B /path/to/input/data:/data \
    -B /output/path/to/store/results/:/results \
    test_model.sif
```

Submit the job

```bash
sbatch <submitfile.sh>
```

See submitted jobs

```bash
squeue --user=<user>
```

The tqdm loop is printed to the `<jobnumber>.err` file.
Therefore, you can continuously check the progress of the training with:

```bash
tail -f <jobnumber>.err
```
