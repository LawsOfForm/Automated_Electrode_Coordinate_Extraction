#!/bin/bash
#SBATCH -J unet_train
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=vision
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -t 72:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=<user-mail>

echo "--------------------------------------------"
echo "###           Start UNet Training        ###"
echo "--------------------------------------------"

echo "Node: $SLURM_JOB_NODELIST"
echo "jobname: $SLURM_JOB_NAME"

echo "Load modules"

module load cuda/12.0.0
module load singularity/3.11.3

echo "Start Container"


model_out="0"
out_base="$HOME/network/output/model_v"
out_dir="${out_base}${model_out}"

while [ -d "${out_dir}" ]; do
    model_out=$((model_out+1))
    out_dir="${out_base}${model_out}"
done

mkdir -p "$out_dir"

# Run the model
# --------------
# `-B` option in singularity is analog to `-v` in docker
#
# The container expects a directory named `data` in
# which the input variables are stored adhering to the
# naming scheme:
#
# - volume: "sub-*/electrode_extraction/ses-*/run-*/petra_.nii.gz"
# - mask: "sub-*/electrode_extraction/ses-*/run-*/petra_masked.nii.gz"


singularity run --nv \
  -B "$HOME/network/input_dir":/data \
  -B "$out_dir":/results \
  model.sif
