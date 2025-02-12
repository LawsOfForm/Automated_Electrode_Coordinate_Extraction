 #!/bin/bash
 #SBATCH -J unet_train
 #SBATCH -N 1
 #SBATCH -n 1
 #SBATCH --partition=vision
 #SBATCH -o %j.out
 #SBATCH -e %j.err
 #SBATCH -t 72:00:00
 #SBATCH --mail-type=end
 #SBATCH --mail-user=filip.niemann@uni-greifswald.de

 echo "--------------------------------------------"
 echo "###           Start UNet Training        ###"
 echo "--------------------------------------------"

 echo "Node: $SLURM_JOB_NODELIST"
 echo "jobname: $SLURM_JOB_NAME"

 echo "Load modules"

 module load cuda/12.0.0

 echo "Start script"

 python3 ~/code/code_for_brain_cluster/model_bc_3d.py
