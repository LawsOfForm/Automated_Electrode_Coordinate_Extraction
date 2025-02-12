# Help run script on brain cluster without singularity

## preprocessing

- zip the folder /media/Data03/Studies/MeMoSLAP/code/code_for_brain_cluster and copy path to braincluster

```bash
cd /media/Data03/Studies/MeMoSLAP/code
zip -r code_for_brain_cluster.zip code_for_brain_cluster
scp code_for_brain_cluster.zip @brain.uni-greifswald.de:code
```

on braincluster unzip

```bash
unzip code_for_brain_cluster.zip
```

## on the brain cluster

- python3 and slurm is preinstalled on cluster
- install all important dependencies with pip
- the requirements.txt list is in code/code_for_brain_cluster

- before installing torch and monai load cuda
- beware torch 2.3.2 is not automatically stable with cuda 12.0.0

```bash
load cuda/12.0.0
load cuda/11.7.0
```

- for torch 2.0.2 cuda 11.7 available

```bash
pip install torch==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu117
```

- then install pytorch and monai

```bash
pip3 install --upgrade pip
pip3 install --no-cache-dir -r ~/code/code_for_brain_cluster/requirements.txt
```

### test if cuda is available

- open python3

```bash
python3
torch.cuda.is_available()
```

```py
import torch
torch.cuda.is_available()
```

### if cuda visible run bashscript

- run bashcript for slurm job with b

```bash
squeue | grep niemannf
```

```bash
sbatch run_model.sh
```

- the job wil run on slurm and you can see the progress with squeue

```bash
squeue | grep niemannf
```

### troubleshooting

#### in bash

- https://stackoverflow.com/questions/62359175/pytorch-says-that-cuda-is-not-available-on-ubuntu
- check cuda availability
  
```bash
nvcc -V
echo $PATH
```

- path problems with library
- use echo $path to control if Library and cuda home folder set correctly

```bash

```

- update bashrc (will lead to error in python3: torch._C import *  # noqa: F403 )

```bash
nano ~\.bashrc
```

```nano
export PATH=/opt/cuda-12.0.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/opt/cuda-12.0.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

- will result in python3 error solvable with

```nano
#export LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```

but now cuda will not be found in python3 via torch

```bash
unset LD_LIBRARY_PATH
```

#### in python3

```py
import torch
from torch.utils.cpp_extension import CUDA_HOME

No CUDA runtime is found, using CUDA_HOME='/opt/cuda-12.0.0'
```

- how to solve ?

```py
import torch
error: ... torch._C import *  # noqa: F403 
```
