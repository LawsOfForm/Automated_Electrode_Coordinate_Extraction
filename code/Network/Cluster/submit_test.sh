#!/usr/bin/env bash
#
# submit_test.sh — SMOKE TEST on the vision-fa partition (usable ~1 day only).
#
# Purpose: verify the FIXED ablation_utils.py (fp32 loss / bf16 autocast /
# grad clipping / mask-transform fix) actually works BEFORE tomorrow's real
# run starts on `vision` at 2026-08-06T09:43:15. This is deliberately cheap:
# one fold per network (not the full 5-fold array), few thousand iterations,
# short walltime. It exists to catch a repeat of the NaN-divergence failure
# (folds 0/1 of job 7714316) in minutes instead of finding out tomorrow.
#
# It does NOT touch the jobs you already have queued on `vision` for
# tomorrow (7733956-7733960) -- different partition, different tag suffix,
# different seed by default, so nothing here can collide with them.
#
# USAGE
#   ./submit_test.sh <network>|all [--dataset HGW|RU] [--iterations N]
#                     [--fold N] [--seed N] [--time HH:MM:SS] [--submit]
#                     [-- <extra py args>]
#
#   <network>: a_baseline | b_no_attention | c_reduced | d_increased | proposed
#              (same names as submit.sh)  or "all" for all five, one fold each.
#
# EXAMPLES
#   ./submit_test.sh d_increased --submit
#       -> the exact network/fold that failed before, fastest way to confirm
#          the fp32-loss fix actually stops the NaN.
#
#   ./submit_test.sh all --submit
#       -> one smoke-test task per network (5 jobs total, not 25), confirms
#          every network at least survives its first few thousand iterations
#          before you commit the vision-fa day to anything bigger.
#
#   ./submit_test.sh proposed --iterations 8000 --time 02:00:00 --submit
#
# WHAT "PASS" LOOKS LIKE
#   tail -f logs/<label>_<jobid>.out
#   - no "[sanity-check] ABORT" line
#   - "[run_experiment] autocast: bf16 (loss always fp32)" printed near the top
#   - Validation Dice > 0 and not stuck at 0.0000 by the last few evals
#   - job ends with "Test Results - Dice: ..." rather than "RUN FAILED"
#
# Check: squeue -p vision-fa -u $USER

set -euo pipefail

ABLATION_DIR="${ABLATION_DIR:-$HOME/ablation}"
SIF="${SIF:-$ABLATION_DIR/pytorch_monai_2.sif}"
MAIL_USER="${MAIL_USER:-niemannf@uni-greifswald.de}"
PARTITION="${PARTITION:-vision-fast}"
GRES="${GRES:-gpu:A100:1}"

# Cheap-but-representative defaults. 5000 iterations comfortably passes the
# point (iter ~4027) where the old code diverged, so this is a real repro of
# the failure, not just an import smoke test. eval-num is lowered so at least
# a couple of validation passes happen inside the short run.
DATASET="HGW"; ITERS=5000; FOLD=0; SEED=9001; TIME="02:00:00"; DO_SUBMIT=0
EXTRA=()

[[ $# -lt 1 ]] && { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
network="$1"; shift

TABLE5=(a_baseline b_no_attention c_reduced d_increased proposed)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)    DATASET="$2"; shift 2 ;;
        --iterations) ITERS="$2"; shift 2 ;;
        --fold)       FOLD="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        --time)       TIME="$2"; shift 2 ;;
        --submit)     DO_SUBMIT=1; shift ;;
        --)           shift; EXTRA=("$@"); break ;;
        *) echo "Unknown option: $1 (use -- before extra python args)" >&2; exit 1 ;;
    esac
done

if [[ "$network" == "all" ]]; then
    echo "Smoke-testing all ${#TABLE5[@]} Table 5 networks (1 fold each) on $PARTITION"
    for n in "${TABLE5[@]}"; do
        echo; echo "=== $n ==="
        "$0" "$n" --dataset "$DATASET" --iterations "$ITERS" --fold "$FOLD" \
            --seed "$SEED" --time "$TIME" $( [[ "$DO_SUBMIT" -eq 1 ]] && echo --submit ) \
            -- "${EXTRA[@]:-}"
    done
    exit 0
fi

case "$network" in
    a_baseline|a)              PY="a_model_unet_baseline.py" ;;
    b_no_attention|b)          PY="b_model_proposed_no_attention.py" ;;
    c_reduced|c)                PY="c_reduced_model_paper_bioarxiv.py" ;;
    d_increased|d|c_increased) PY="c_increased_model_paper_bioarxiv.py" ;;
    proposed|c_normal)         PY="c_normal_model_paper_bioarxiv.py" ;;
    *) echo "ERROR: unknown network '$network'." >&2
       echo "Valid: a_baseline b_no_attention c_reduced d_increased proposed (or 'all')" >&2
       exit 1 ;;
esac

if [[ ! -f "$ABLATION_DIR/$PY" ]]; then
    echo "ERROR: $ABLATION_DIR/$PY not found -- copy it to the cluster first." >&2
    exit 1
fi

# Tag is unmistakably a test run and cannot collide with tomorrow's real jobs
# (those use suffix "_<network>" only, seed 1001, on partition vision).
LABEL="${network}_${DATASET}_smoketest_f${FOLD}_seed${SEED}"
TAG_SUFFIX="_${network}_smoketest"

mkdir -p "$ABLATION_DIR/slurm" "$ABLATION_DIR/logs"
OUT="$ABLATION_DIR/slurm/slurm_${LABEL}.sh"

cat > "$OUT" <<EOF
#!/bin/bash
#SBATCH -J smoke_${LABEL}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t ${TIME}
#SBATCH -o ${ABLATION_DIR}/logs/${LABEL}_%j.out
#SBATCH -e ${ABLATION_DIR}/logs/${LABEL}_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${MAIL_USER}

set -euo pipefail
echo "SMOKE TEST  network=${network}  dataset=${DATASET}  fold=${FOLD}  iters=${ITERS}"
echo "Node: \$SLURM_JOB_NODELIST   Start: \$(date)"

module load singularity/3.11.3
nvidia-smi -L || { echo "ERROR: no GPU visible"; exit 1; }

singularity exec --nv "${SIF}" \\
    python3 "${ABLATION_DIR}/${PY}" \\
        --dataset ${DATASET} \\
        --seed ${SEED} \\
        --tag-suffix ${TAG_SUFFIX} \\
        --folds 5 --fold ${FOLD} \\
        --max-iterations ${ITERS} \\
        --eval-num 250 \\
        --sanity-probation-iters ${ITERS} \\
        --optimizer adam_paperbc3d --lr 5e-4 \\
        ${EXTRA[*]:-}

echo "End: \$(date)"
echo
echo "PASS/FAIL check:"
grep -q '\\[sanity-check\\] ABORT' "${ABLATION_DIR}/logs/${LABEL}_\${SLURM_JOB_ID}.out" \\
    && echo "  FAIL: sanity check aborted -- see log" \\
    || echo "  no abort raised"
EOF

chmod +x "$OUT"
echo "Wrote $OUT"
echo "  network    : $network -> $PY"
echo "  partition  : $PARTITION   time: $TIME"
echo "  dataset    : $DATASET   fold: $FOLD (of 5)   seed: $SEED   iters: $ITERS"
echo "  tag suffix : $TAG_SUFFIX   (cannot collide with tomorrow's _${network} run)"

if [[ "$DO_SUBMIT" -eq 1 ]]; then
    SB_OUT="$(sbatch "$OUT")"
    echo "$SB_OUT"
else
    echo
    echo "Review it, then:"
    echo "    sbatch $OUT"
fi
