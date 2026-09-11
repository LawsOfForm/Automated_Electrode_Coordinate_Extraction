#!/usr/bin/env bash
#
# submit.sh — ONE script to launch any ablation network, on either dataset,
# as a single run or as a subject-wise 5-fold CV array job.
#
# Output collisions are impossible. ablation_utils.py auto-builds the tag from
#   <dataset>_<cv-fold|split>_seed<seed>
# and this script always adds  --tag-suffix _<network>. The network name is
# essential: c_normal and c_increased BOTH hardcode
# EXPERIMENT_TAG="paper_attention_unet", so without it they would overwrite each
# other's checkpoint, report, TensorBoard dir and per-case CSV.
# Final tag e.g.:  paper_attention_unet_HGW_cv5f2_seed1001_c_increased
#
# USAGE
#   ./submit.sh <network> [--dataset HGW|RU] [--cv|--final] [--seed N] [--submit]
#                          [-- <extra py args>]
#
#   --submit  runs sbatch immediately AND appends one row per array task to
#             ~/ablation/jobs.csv (append-only job registry). Without it the
#             script only writes the SLURM file for you to review.
#
#   --final   trains the ONE deployable model on every participant (minus a
#             small validation slice for checkpoint selection), no test set.
#             Run this AFTER --cv has told you which network to ship and what
#             performance to expect. Mutually exclusive with --cv.
#
#   <network>:  one of the manuscript Table 5 rows --
#                 a_baseline      (a) Standard U-Net,          64-512, no attention
#                 b_no_attention  (b) Proposed depths,         32-512, no attention
#                 c_reduced       (c) Attention U-Net reduced, 16-256
#                 d_increased     (d) Attention U-Net increased, 48-768
#                 proposed            Proposed Attention U-Net, 32-512
#               or "all" to launch all five, or "segresnet" (not in Table 5).
#
# EXAMPLES
#   ./submit.sh c_increased                       # single split, HGW
#   ./submit.sh c_increased --cv                  # 5-fold CV -> one array job, 5 tasks
#   ./submit.sh c_normal --dataset RU --cv
#   ./submit.sh c_increased --seed 2002
#   ./submit.sh c_increased --cv -- --optimizer adam_paperbc3d --lr 5e-4
#
#   Launch the whole Table 5 ablation as 5-fold CV (5 networks x 5 folds = 25 tasks):
#     ./submit.sh all --cv --submit
#
#   Same on the RU dataset:
#     ./submit.sh all --dataset RU --cv --submit
#
# Check:  squeue -u $USER        Logs:  tail -f ~/ablation/logs/<tag>_*.out

set -euo pipefail

ABLATION_DIR="${ABLATION_DIR:-$HOME/ablation}"
SIF="${SIF:-$ABLATION_DIR/pytorch_monai_2.sif}"
MAIL_USER="${MAIL_USER:-niemannf@uni-greifswald.de}"
DEFAULT_OPT="${DEFAULT_OPT:---optimizer adam_paperbc3d --lr 5e-4}"
PARTITION="${PARTITION:-vision}"
GRES="${GRES:-gpu:A100:1}"

DATASET="HGW"; CV=0; FINAL=0; SEED=1001; DO_SUBMIT=0; EXTRA=()

[[ $# -lt 1 ]] && { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
network="$1"; shift

# The five networks of manuscript Table 5, in table order.
TABLE5=(a_baseline b_no_attention c_reduced d_increased proposed)

if [[ "$network" == "all" ]]; then
    echo "Launching all ${#TABLE5[@]} Table 5 networks with: $*"
    for n in "${TABLE5[@]}"; do
        echo; echo "=== $n ==="
        "$0" "$n" "$@" || echo "  (skipped: $n)"
    done
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --cv)      CV=1; shift ;;
        --final)   FINAL=1; shift ;;
        --submit)  DO_SUBMIT=1; shift ;;
        --seed)    SEED="$2"; shift 2 ;;
        --)        shift; EXTRA=("$@"); break ;;
        *) echo "Unknown option: $1 (use -- before extra python args)" >&2; exit 1 ;;
    esac
done

# Network names follow the manuscript's Table 5 rows, so a result file can be
# traced to a table row without a lookup. Legacy script names still work.
case "$network" in
    a_baseline|a)                 PY="a_model_unet_baseline.py" ;;
    b_no_attention|b)             PY="b_model_proposed_no_attention.py" ;;
    c_reduced|c)                  PY="c_reduced_model_paper_bioarxiv.py" ;;
    d_increased|d|c_increased)    PY="c_increased_model_paper_bioarxiv.py" ;;
    proposed|c_normal)            PY="c_normal_model_paper_bioarxiv.py" ;;
    segresnet)                    PY="d_model_segresnet.py" ;;
    *) echo "ERROR: unknown network '$network'." >&2
       echo "Table 5 rows:  a_baseline  b_no_attention  c_reduced  d_increased  proposed" >&2
       echo "Extra:         segresnet   (not in Table 5)" >&2
       exit 1 ;;
esac

if [[ ! -f "$ABLATION_DIR/$PY" ]]; then
    echo "ERROR: $ABLATION_DIR/$PY not found -- copy it to the cluster first." >&2
    exit 1
fi

[[ "$DATASET" == "HGW" || "$DATASET" == "RU" ]] || { echo "ERROR: --dataset must be HGW or RU" >&2; exit 1; }

# The tag mirrors what ablation_utils builds, so log filenames match the outputs.
if [[ "$CV" -eq 1 && "$FINAL" -eq 1 ]]; then
    echo "ERROR: --cv and --final are mutually exclusive (--final trains ONE" >&2
    echo "       deployable model on all data; that's what --cv already told you" >&2
    echo "       to expect the performance of)." >&2
    exit 1
fi

if [[ "$FINAL" -eq 1 ]]; then
    LABEL="${network}_${DATASET}_FINAL_seed${SEED}"
    ARRAY_DIRECTIVE=""
    FOLD_ARGS="--split-mode final"
    LOGSUFFIX="%j"
elif [[ "$CV" -eq 1 ]]; then
    LABEL="${network}_${DATASET}_cv5_seed${SEED}"
    ARRAY_DIRECTIVE="#SBATCH --array=0-4"
    FOLD_ARGS='--folds 5 --fold $SLURM_ARRAY_TASK_ID'
    LOGSUFFIX="%A_%a"
else
    LABEL="${network}_${DATASET}_single_seed${SEED}"
    ARRAY_DIRECTIVE=""
    FOLD_ARGS=""
    LOGSUFFIX="%j"
fi

mkdir -p "$ABLATION_DIR/slurm" "$ABLATION_DIR/logs"
OUT="$ABLATION_DIR/slurm/slurm_${LABEL}.sh"

cat > "$OUT" <<EOF
#!/bin/bash
#SBATCH -J ${LABEL}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 72:00:00
#SBATCH -o ${ABLATION_DIR}/logs/${LABEL}_${LOGSUFFIX}.out
#SBATCH -e ${ABLATION_DIR}/logs/${LABEL}_${LOGSUFFIX}.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL_USER}
${ARRAY_DIRECTIVE}

set -euo pipefail
echo "Job \$SLURM_JOB_ID  network=${network}  dataset=${DATASET}  fold=\${SLURM_ARRAY_TASK_ID:-none}"
echo "Node: \$SLURM_JOB_NODELIST   Start: \$(date)"

module load singularity/3.11.3
nvidia-smi -L || { echo "ERROR: no GPU visible"; exit 1; }

singularity exec --nv "${SIF}" \\
    python3 "${ABLATION_DIR}/${PY}" \\
        --dataset ${DATASET} \\
        --seed ${SEED} \\
        --tag-suffix _${network} \\
        ${FOLD_ARGS} \\
        ${DEFAULT_OPT} ${EXTRA[*]:-}

echo "End: \$(date)"
EOF

chmod +x "$OUT"

# ---------------------------------------------------------------------------
# Job registry (append-only). One row per array task, written at submit time.
# Append-only means concurrent submits can never corrupt it; the mutable view
# (status.csv) is derived from this file by track.py and can be regenerated.
# ---------------------------------------------------------------------------
REG="$ABLATION_DIR/jobs.csv"

register_rows() {                       # $1 = job id ("" if not submitted)
    local jid="$1" now folds_v fold_v key logf tagglob
    [[ -f "$REG" ]] || echo "submitted_at,job_key,job_id,array_task,network,dataset,folds,fold,seed,py_script,slurm_script,log_out,tag_glob" > "$REG"
    now="$(date -Is)"
    if [[ "$CV" -eq 1 ]]; then folds_v=5; else folds_v=0; fi
    for fold_v in $( [[ "$CV" -eq 1 ]] && echo 0 1 2 3 4 || echo 0 ); do
        if [[ "$CV" -eq 1 ]]; then
            key="${jid:-PENDING}_${fold_v}"
            logf="$ABLATION_DIR/logs/${LABEL}_${jid:-UNKNOWN}_${fold_v}.out"
            tagglob="*_${DATASET}_cv5f${fold_v}_seed${SEED}_${network}"
        else
            key="${jid:-PENDING}"
            logf="$ABLATION_DIR/logs/${LABEL}_${jid:-UNKNOWN}.out"
            tagglob="*_${DATASET}_seed${SEED}_${network}"
        fi
        echo "$now,$key,${jid:-},$( [[ "$CV" -eq 1 ]] && echo "$fold_v" || echo ),$network,$DATASET,$folds_v,$fold_v,$SEED,$PY,$OUT,$logf,$tagglob" >> "$REG"
    done
}

if [[ "$DO_SUBMIT" -eq 1 ]]; then
    SB_OUT="$(sbatch "$OUT")"           # e.g. "Submitted batch job 7714316"
    echo "$SB_OUT"
    JOBID="$(grep -oE '[0-9]+$' <<<"$SB_OUT")"
    register_rows "$JOBID"
    echo "Registered $( [[ "$CV" -eq 1 ]] && echo 5 || echo 1 ) row(s) in $REG"
else
    echo "Wrote $OUT (not submitted)"
fi

echo "  network : $network -> $PY"
echo "  dataset : $DATASET     seed: $SEED"
if [[ "$FINAL" -eq 1 ]]; then
    echo "  mode    : FINAL DEPLOYABLE MODEL -- trained on all participants, no test set"
    echo "            report the earlier --cv run's mean as this model's expected performance"
elif [[ "$CV" -eq 1 ]]; then
    echo "  mode    : 5-fold subject-wise CV (array 0-4, 5 tasks)"
else
    echo "  mode    : single subject-wise split"
fi
if [[ "$DO_SUBMIT" -eq 0 ]]; then
    echo
    echo "Review it, then either:"
    echo "    sbatch $OUT                 # submit only"
    echo "    $0 $network ... --submit    # submit AND register in jobs.csv"
fi
