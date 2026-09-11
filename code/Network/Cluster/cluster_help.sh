#!/usr/bin/env bash
#
# cluster_help.sh — one-shot deploy of the ablation study to the
# Greifswald "brain" HPC cluster.
#
# What it does:
#   1. Creates the remote directory tree (~/ablation/{slurm,logs,...}).
#   2. scp's ONLY the files needed to train (the .py scripts + the
#      Singularity .def). It deliberately skips debug_images/, models/,
#      runs/, ablation_results/ etc. — those are gigabytes and are NOT
#      needed on the cluster to start training.
#   3. Writes the SLURM submit scripts on the cluster, pre-wired for the
#      A100 GPU partition.
#   4. Prints the exact next commands (build image once, then sbatch).
#
# After running this, on the cluster you only have to:
#       cd ~/ablation
#       module load singularity/3.11.3
#       singularity build --fakeroot pytorch_monai.sif pytorch_monai.def   # once
#       sbatch slurm/slurm_c_increased.sh                                  # train
#
# Usage:
#       ./cluster_help.sh                 # uses defaults below
#       ./cluster_help.sh -u niemannf -H brain.uni-greifswald.de
#       LOCAL_SRC=/path/to/3D_local ./cluster_help.sh
#
# run on cluster without password
#Best option: SSH keys (no password stored at all)
#This is the standard approach on HPC clusters and what I'd recommend. You generate a key pair once, put the public half on the cluster, and from then #on ssh/scp/rsync authenticate with no password and nothing secret sitting in a file you have to protect.
#bash# 1. Generate a key (once). Press enter for default path; set a passphrase if you like.
#ssh-keygen -t ed25519 -C "niemannf@brain-cluster"

# 2. Copy the public key to the cluster (this asks for your password ONE time).
#ssh-copy-id niemannf@brain.uni-greifswald.de

# 3. Done — now this works with no password:
#ssh niemannf@brain.uni-greifswald.de


set -euo pipefail

# ---------------------------------------------------------------------------
# Config (override via flags or environment variables)
# ---------------------------------------------------------------------------
USER_NAME="${USER_NAME:-niemannf}"
CLUSTER_HOST="${CLUSTER_HOST:-brain.uni-greifswald.de}"   # you asked for this host
REMOTE_DIR="${REMOTE_DIR:-ablation}"                       # ~/ablation on the cluster
LOCAL_SRC="${LOCAL_SRC:-.}"                                # where your .py files live locally
DATASET_ROOT="${DATASET_ROOT:-/media/data04/Automatic_Electrode_extraction/Dataset/dataset_RU}"
MAIL_USER="${MAIL_USER:-${USER_NAME}@uni-greifswald.de}"
SSH_OPTS="${SSH_OPTS:-}"     # e.g. SSH_OPTS='-J jumphost' for ssh -J, or '-p 2222'

while getopts "u:H:d:s:r:m:h" opt; do
    case "$opt" in
        u) USER_NAME="$OPTARG" ;;
        H) CLUSTER_HOST="$OPTARG" ;;
        d) DATASET_ROOT="$OPTARG" ;;
        s) LOCAL_SRC="$OPTARG" ;;
        r) REMOTE_DIR="$OPTARG" ;;
        m) MAIL_USER="$OPTARG" ;;
        h)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option. Run with -h for help." >&2; exit 1 ;;
    esac
done

REMOTE="${USER_NAME}@${CLUSTER_HOST}"
# shellcheck disable=SC2086
SSH() { ssh $SSH_OPTS "$REMOTE" "$@"; }
# shellcheck disable=SC2086
SCP() { scp $SSH_OPTS "$@"; }

echo "============================================================"
echo " Deploying ablation study to the cluster"
echo "   remote      : $REMOTE"
echo "   remote dir  : ~/$REMOTE_DIR"
echo "   local source: $LOCAL_SRC"
echo "   dataset root: $DATASET_ROOT"
echo "============================================================"

# ---------------------------------------------------------------------------
# 0. Sanity check: are the files we need actually present locally?
# ---------------------------------------------------------------------------
# Core code needed for training. ablation_utils.py + gpu_batch_helper.py are
# imported by the model scripts, so they are mandatory.
CORE_FILES=(
    ablation_utils.py
    gpu_batch_helper.py
    a_model_unet_baseline.py
    b_model_proposed_no_attention.py
    c_model_channel_depth.py
    d_model_segresnet.py
    aggregate_ablation_results.py
    dataset.py
    build_net.py
)
# Optional-but-nice extras (copied if present, skipped quietly if not).
OPTIONAL_FILES=(
    README.md
    model.py
)

echo
echo "--> Checking which core files exist under: $LOCAL_SRC"
present=()
missing=()
for f in "${CORE_FILES[@]}"; do
    if [[ -f "$LOCAL_SRC/$f" ]]; then
        present+=("$f")
    else
        missing+=("$f")
    fi
done
for f in "${present[@]}"; do echo "      found : $f"; done
for f in "${missing[@]}"; do echo "      MISSING: $f (will skip — fix LOCAL_SRC if unexpected)"; done

if [[ ${#present[@]} -eq 0 ]]; then
    echo
    echo "ERROR: none of the expected .py files were found in '$LOCAL_SRC'." >&2
    echo "       Run this from inside your 3D_local folder, or pass -s /path/to/3D_local" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Create the remote directory tree
# ---------------------------------------------------------------------------
echo
echo "--> Creating remote directory tree under ~/$REMOTE_DIR"
SSH "mkdir -p ~/$REMOTE_DIR/slurm ~/$REMOTE_DIR/logs ~/$REMOTE_DIR/ablation_results"

# ---------------------------------------------------------------------------
# 2. Copy the code (NOT the multi-GB images / checkpoints / tfevents)
# ---------------------------------------------------------------------------
echo
echo "--> Copying core code files"
COPY_LIST=()
for f in "${present[@]}"; do COPY_LIST+=("$LOCAL_SRC/$f"); done
for f in "${OPTIONAL_FILES[@]}"; do
    [[ -f "$LOCAL_SRC/$f" ]] && COPY_LIST+=("$LOCAL_SRC/$f")
done

# Prefer rsync (restartable, shows progress); fall back to scp.
if command -v rsync >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    rsync -avh --progress -e "ssh $SSH_OPTS" "${COPY_LIST[@]}" "$REMOTE:~/$REMOTE_DIR/"
else
    SCP "${COPY_LIST[@]}" "$REMOTE:~/$REMOTE_DIR/"
fi

# Copy the Singularity definition file if you keep one alongside the code
# (e.g. in a Cluster/ subfolder). Otherwise we generate one below.
DEF_SRC=""
for cand in "$LOCAL_SRC/pytorch_monai.def" "$LOCAL_SRC/Cluster/pytorch_monai.def"; do
    [[ -f "$cand" ]] && DEF_SRC="$cand" && break
done
if [[ -n "$DEF_SRC" ]]; then
    echo "--> Copying existing Singularity def: $DEF_SRC"
    SCP "$DEF_SRC" "$REMOTE:~/$REMOTE_DIR/pytorch_monai.def"
fi

# The prebuilt .sif CANNOT be built on the cluster, so we always push it.
# It lives in the Cluster/ subfolder of 3D_local (or alongside the code).
SIF_SRC=""
for cand in "$LOCAL_SRC/pytorch_monai.sif" "$LOCAL_SRC/Cluster/pytorch_monai.sif"; do
    [[ -f "$cand" ]] && SIF_SRC="$cand" && break
done
if [[ -z "$SIF_SRC" ]]; then
    echo
    echo "ERROR: pytorch_monai.sif not found locally." >&2
    echo "       Looked in:" >&2
    echo "         $LOCAL_SRC/pytorch_monai.sif" >&2
    echo "         $LOCAL_SRC/Cluster/pytorch_monai.sif" >&2
    echo "       Point -s at your 3D_local folder, or set SIF_SRC=/path/to/pytorch_monai.sif" >&2
    exit 1
fi

echo
echo "--> Pushing prebuilt Singularity image: $SIF_SRC ($(du -h "$SIF_SRC" | cut -f1))"
echo "    (several GB — this can take 10-30 min depending on your link)"
# rsync is restartable, so a network hiccup mid-transfer won't force a
# full re-send. Resumes from where it left off if you re-run the script.
if command -v rsync >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    rsync -avh --progress --partial --inplace -e "ssh $SSH_OPTS" \
        "$SIF_SRC" "$REMOTE:~/$REMOTE_DIR/pytorch_monai.sif"
else
    SCP "$SIF_SRC" "$REMOTE:~/$REMOTE_DIR/pytorch_monai.sif"
fi
echo "    image in place: ~/$REMOTE_DIR/pytorch_monai.sif"

# ---------------------------------------------------------------------------
# 3. Generate the Singularity .def on the cluster if we didn't copy one
# ---------------------------------------------------------------------------
echo
echo "--> Ensuring a Singularity definition exists on the cluster"
SSH "bash -s" <<REMOTE_DEF
set -euo pipefail
cd ~/$REMOTE_DIR
if [[ ! -f pytorch_monai.def ]]; then
cat > pytorch_monai.def <<'DEF'
Bootstrap: docker
From: nvcr.io/nvidia/pytorch:24.10-py3
# Ships PyTorch 2.5 + CUDA 12.6 + cuDNN/NCCL, built for NVIDIA GPUs (A100 OK).

%post
    pip install --no-cache-dir --no-deps monai==1.4.0
    pip install --no-cache-dir nibabel tqdm tensorboard matplotlib einops

%environment
    export PYTHONWARNINGS="ignore::DeprecationWarning"
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1

%labels
    Maintainer $USER_NAME
    Purpose    Ablation study for the Automated Electrode Extraction paper
DEF
echo "    wrote pytorch_monai.def"
else
echo "    pytorch_monai.def already present — leaving it alone"
fi
REMOTE_DEF

# ---------------------------------------------------------------------------
# 4. Write the SLURM submit scripts on the cluster (A100-targeted)
# ---------------------------------------------------------------------------
echo
echo "--> Writing SLURM scripts into ~/$REMOTE_DIR/slurm"
SSH "REMOTE_DIR='$REMOTE_DIR' DATASET_ROOT='$DATASET_ROOT' MAIL_USER='$MAIL_USER' bash -s" <<'REMOTE_SLURM'
set -euo pipefail
cd ~/"$REMOTE_DIR"

# Emit one SLURM script. Args: <file> <jobname> <python invocation line(s)>
make_slurm() {
    local file="$1"; local jobname="$2"; shift 2
    local pyargs="$*"
    cat > "slurm/$file" <<EOF
#!/bin/bash
#SBATCH -J $jobname
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1                  # request an A100 specifically
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$MAIL_USER

set -euo pipefail

echo "Node:    \$SLURM_JOB_NODELIST"
echo "Job ID:  \$SLURM_JOB_ID"
echo "Start:   \$(date -Is)"
echo "PWD:     \$PWD"

cd \$HOME/$REMOTE_DIR
module load singularity/3.11.3
echo "Singularity: \$(singularity --version)"

DATASET_ROOT=$DATASET_ROOT

singularity exec --nv \\
    -B "\$DATASET_ROOT":"\$DATASET_ROOT":ro \\
    pytorch_monai.sif \\
    $pyargs

echo "End: \$(date -Is)"
EOF
    chmod +x "slurm/$file"
    echo "    wrote slurm/$file"
}

# (a) standard U-Net baseline
make_slurm slurm_a_baseline.sh abl_a_unet \
'python a_model_unet_baseline.py \
        --dataset-root "$DATASET_ROOT" \
        --num-workers 2'

# (b) proposed without attention
make_slurm slurm_b_no_attention.sh abl_b_no_attn \
'python b_model_proposed_no_attention.py \
        --dataset-root "$DATASET_ROOT" \
        --num-workers 2'

# (c) proposed at reduced channel depth
make_slurm slurm_c_reduced.sh abl_c_reduced \
'python c_model_channel_depth.py \
        --variant reduced \
        --dataset-root "$DATASET_ROOT" \
        --num-workers 2'

# (c) proposed at normal/published channel depth
make_slurm slurm_c_normal.sh abl_c_normal \
'python c_model_channel_depth.py \
        --variant normal \
        --dataset-root "$DATASET_ROOT" \
        --num-workers 2'

# (c) proposed at increased channel depth — heaviest; A100 gives the headroom
make_slurm slurm_c_increased.sh abl_c_increased \
'python c_model_channel_depth.py \
        --variant increased \
        --dataset-root "$DATASET_ROOT" \
        --num-workers 2'

echo
echo "    SLURM scripts in place:"
ls -1 slurm/
REMOTE_SLURM

# ---------------------------------------------------------------------------
# 5. Final instructions
# ---------------------------------------------------------------------------
cat <<EOF

============================================================
 Done. Everything is on the cluster under ~/$REMOTE_DIR
============================================================

Next steps (on the cluster):

  ssh ${SSH_OPTS} $REMOTE
  cd ~/$REMOTE_DIR
  module load singularity/3.11.3

  # The prebuilt image is already here (pushed by this script) — no build
  # needed. Optional one-line smoke test:
  singularity exec pytorch_monai.sif python -c "import torch, monai; print(torch.__version__, monai.__version__)"

  # Then just submit the training job(s):
  sbatch slurm/slurm_a_baseline.sh
  sbatch slurm/slurm_b_no_attention.sh
  sbatch slurm/slurm_c_reduced.sh
  sbatch slurm/slurm_c_normal.sh
  sbatch slurm/slurm_c_increased.sh

  squeue -u \$USER          # watch the queue
  tail -f logs/<job>.out    # watch progress

Notes:
  * The SLURM scripts request an A100 via  --gres=gpu:a100:1.
    If the cluster doesn't label A100s that way, check with
      sinfo -o "%P %G"
    and adjust the --gres line (e.g. plain  --gres=gpu:1 ).
  * Dataset path baked into the scripts:
      $DATASET_ROOT
    Override per-job by editing DATASET_ROOT in the slurm/*.sh file.
  * The host you asked for is $CLUSTER_HOST. If login fails, the guide
    also lists  apphub.uni-greifswald.de  /  login-a.brain.uni-greifswald.de.
    Re-run with  -H apphub.uni-greifswald.de  if needed.
EOF
