#!/usr/bin/env bash
#
# make_slurm.sh — generate a ready-to-submit SLURM script for ONE ablation
# network, with a unique --tag-suffix so concurrent jobs never overwrite each
# other's checkpoint / report / TensorBoard files.
#
# Why this exists:
#   c_normal_model_paper_bioarxiv.py and c_increased_model_paper_bioarxiv.py
#   BOTH hardcode EXPERIMENT_TAG="paper_attention_unet". Run together, they
#   clobber paper_attention_unet_best_metric_model.pth and *_report.md.
#   The --tag-suffix flag (added to ablation_utils.py) fixes this; this script
#   wires a sensible suffix automatically per network.
#
# Usage:
#   ./make_slurm.sh <network> [extra args passed to the python script ...]
#
#   <network> is one of:
#     c_normal      -> c_normal_model_paper_bioarxiv.py     (suffix _normal)
#     c_increased   -> c_increased_model_paper_bioarxiv.py  (suffix _increased)
#     segresnet     -> d_model_segresnet.py                 (suffix from --init-filters)
#     a_baseline    -> a_model_unet_baseline.py             (suffix _a_baseline)
#     b_no_attention-> b_model_proposed_no_attention.py     (suffix _b_no_attn)
#
# Examples:
#   ./make_slurm.sh c_increased
#   ./make_slurm.sh c_normal --optimizer adam_paperbc3d --lr 5e-4
#   ./make_slurm.sh segresnet --init-filters 16 --auto-tune-batch
#   ./make_slurm.sh c_increased --seed 2002 --tag-suffix _increased_seed2002
#
# It writes slurm/slurm_<suffix>.sh and prints the sbatch command. It does
# NOT submit — you review then sbatch yourself.

set -euo pipefail

ABLATION_DIR="${ABLATION_DIR:-/home/niemannf/ablation}"
SIF="${SIF:-$ABLATION_DIR/pytorch_monai_2.sif}"
DATASET_ROOT="${DATASET_ROOT:-/home/niemannf/Dataset/dataset_RU}"
MAIL_USER="${MAIL_USER:-niemannf@uni-greifswald.de}"
DEFAULT_OPT="${DEFAULT_OPT:---optimizer adam_paperbc3d --lr 5e-4}"   # the stable recipe

if [[ $# -lt 1 ]]; then
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

network="$1"; shift
extra_args=("$@")

# Map the friendly name -> (python script, default tag suffix, job name).
case "$network" in
    c_normal)
        PY="c_normal_model_paper_bioarxiv.py"; SUFFIX="_normal";    JOB="abl_c_normal" ;;
    c_increased)
        PY="c_increased_model_paper_bioarxiv.py"; SUFFIX="_increased"; JOB="abl_c_increased" ;;
    segresnet)
        PY="d_model_segresnet.py"; SUFFIX="";       JOB="abl_d_segresnet" ;;
    a_baseline)
        PY="a_model_unet_baseline.py"; SUFFIX="_a_baseline"; JOB="abl_a_baseline" ;;
    b_no_attention)
        PY="b_model_proposed_no_attention.py"; SUFFIX="_b_no_attn"; JOB="abl_b_no_attn" ;;
    *)
        echo "ERROR: unknown network '$network'." >&2
        echo "Valid: c_normal c_increased segresnet a_baseline b_no_attention" >&2
        exit 1 ;;
esac

# If the user passed their own --tag-suffix in extra_args, don't add ours
# (avoid a double suffix). Detect it.
user_set_suffix=0
for a in "${extra_args[@]:-}"; do
    [[ "$a" == "--tag-suffix" || "$a" == --tag-suffix=* ]] && user_set_suffix=1
done

# For segresnet, the script's own tag already encodes init_filters, so a
# suffix is optional; we leave SUFFIX empty unless the user sets one.
suffix_args=()
if [[ "$user_set_suffix" -eq 0 && -n "$SUFFIX" ]]; then
    suffix_args=(--tag-suffix "$SUFFIX")
fi

# The label used in filenames / logs. If the user supplied their own
# --tag-suffix, use that for the filename so it's unique; else our default;
# else the network name.
user_suffix_val=""
prev=""
for a in "${extra_args[@]:-}"; do
    if [[ "$a" == --tag-suffix=* ]]; then user_suffix_val="${a#--tag-suffix=}"; fi
    if [[ "$prev" == "--tag-suffix" ]]; then user_suffix_val="$a"; fi
    prev="$a"
done
if [[ -n "$user_suffix_val" ]]; then
    label="${user_suffix_val#_}"
else
    label="${SUFFIX:-_$network}"
    label="${label#_}"   # strip leading underscore for the filename
fi

OUT="slurm/slurm_${label}.sh"
mkdir -p "$(dirname "$ABLATION_DIR/$OUT")" 2>/dev/null || true
mkdir -p slurm

cat > "$OUT" <<EOF
#!/bin/bash
#SBATCH -J ${JOB}
#SBATCH --partition=vision
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1                       # capital A100 — case-sensitive
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 72:00:00
#SBATCH -o ${ABLATION_DIR}/logs/${label}_%j.out
#SBATCH -e ${ABLATION_DIR}/logs/${label}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL_USER}

set -euo pipefail

echo "Job \$SLURM_JOB_ID  network=${network}  on \$SLURM_JOB_NODELIST  GPUs=\${CUDA_VISIBLE_DEVICES:-none}"
echo "Start: \$(date)"

module load singularity/3.11.3

ABLATION_DIR="${ABLATION_DIR}"
SIF="${SIF}"
PY_SCRIPT="\$ABLATION_DIR/${PY}"
DATASET_ROOT="${DATASET_ROOT}"

nvidia-smi -L || { echo "ERROR: no GPU visible"; exit 1; }

singularity exec --nv "\$SIF" \\
    python3 "\$PY_SCRIPT" \\
        --dataset-root "\$DATASET_ROOT" \\
        ${DEFAULT_OPT} \\
        ${suffix_args[*]:-} ${extra_args[*]:-}

echo "End: \$(date)"
EOF

chmod +x "$OUT"

echo "Wrote $OUT"
echo "  network      : $network -> $PY"
echo "  tag suffix   : ${suffix_args[*]:-<none / user-supplied>}"
echo "  extra args   : ${extra_args[*]:-<none>}"
echo
echo "Review it, then submit with:"
echo "    sbatch $OUT"
