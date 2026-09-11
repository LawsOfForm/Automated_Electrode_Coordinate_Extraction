#!/usr/bin/env bash
#
# submit_loss_test.sh — is DiceFocalLoss causing the collapses?
#
# THE QUESTION
#   Under the current loss, DiceFocalLoss(gamma=2.5, include_background=False),
#   7 of 24 completed CV folds produced at least one zero-Dice case (complete
#   prediction collapse -> inf Hausdorff), across EVERY architecture, and one
#   fold (c_reduced f3) stalled at a flat loss of 0.30 and never learned.
#   Every variant also sits ~0.10 Dice below the previously published 0.72.
#   That pattern is systematic, not architectural, which points at the
#   objective rather than the networks.
#
# THE DESIGN
#   Same architecture, same folds, same seed, same optimiser/LR -- ONLY the
#   loss changes. Because the folds are seed-locked, each run pairs exactly
#   with the corresponding run from the existing --cv results, so the
#   comparison is paired per fold rather than between two independent means.
#
#   Default is the two folds that actually collapsed for the proposed model
#   (f2 and f4). If the collapses disappear there, run all five.
#
# USAGE
#   ./submit_loss_test.sh                          # proposed, dicece, folds 2 and 4
#   ./submit_loss_test.sh --network proposed --loss dicece --folds "0 1 2 3 4"
#   ./submit_loss_test.sh --network c_reduced --folds 3     # the fold that never learned
#   ./submit_loss_test.sh --loss diceloss                   # the other candidate
#   ./submit_loss_test.sh --submit
#
# READING THE RESULT
#   python3 compare_loss.py
#   -> pairs each new fold against the dicefocal fold of the same index and
#      reports the change in Dice and in the number of zero-Dice cases.
#
#   The decisive metric is NOT mean Dice, it is the COUNT OF ZERO-DICE CASES.
#   If those go to zero, the loss was the problem and the whole ablation
#   should be re-run under the new loss before anything goes in the paper.

set -euo pipefail

ABLATION_DIR="${ABLATION_DIR:-$HOME/ablation}"
SIF="${SIF:-$ABLATION_DIR/pytorch_monai_2.sif}"
MAIL_USER="${MAIL_USER:-niemannf@uni-greifswald.de}"
PARTITION="${PARTITION:-vision-fast}"
GRES="${GRES:-gpu:A100:1}"
TIME="${TIME:-12:00:00}"

NETWORK="proposed"; LOSS="dicece"; FOLDS="2 4"; DATASET="HGW"; SEED=1001
DO_SUBMIT=0; EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --network) NETWORK="$2"; shift 2 ;;
        --loss)    LOSS="$2"; shift 2 ;;
        --folds)   FOLDS="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --seed)    SEED="$2"; shift 2 ;;
        --time)    TIME="$2"; shift 2 ;;
        --submit)  DO_SUBMIT=1; shift ;;
        --)        shift; EXTRA=("$@"); break ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

case "$NETWORK" in
    a_baseline)     PY="a_model_unet_baseline.py" ;;
    b_no_attention) PY="b_model_proposed_no_attention.py" ;;
    c_reduced)      PY="c_reduced_model_paper_bioarxiv.py" ;;
    d_increased)    PY="c_increased_model_paper_bioarxiv.py" ;;
    proposed)       PY="c_normal_model_paper_bioarxiv.py" ;;
    *) echo "ERROR: unknown network '$NETWORK'" >&2; exit 1 ;;
esac
[[ -f "$ABLATION_DIR/$PY" ]] || { echo "ERROR: $ABLATION_DIR/$PY not found" >&2; exit 1; }

mkdir -p "$ABLATION_DIR/slurm" "$ABLATION_DIR/logs"

for FOLD in $FOLDS; do
    LABEL="${NETWORK}_${DATASET}_${LOSS}_f${FOLD}_seed${SEED}"
    OUT="$ABLATION_DIR/slurm/slurm_${LABEL}.sh"
    cat > "$OUT" <<EOF
#!/bin/bash
#SBATCH -J loss_${LABEL}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t ${TIME}
#SBATCH -o ${ABLATION_DIR}/logs/${LABEL}_%j.out
#SBATCH -e ${ABLATION_DIR}/logs/${LABEL}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL_USER}

set -euo pipefail
echo "LOSS TEST  network=${NETWORK}  loss=${LOSS}  fold=${FOLD}  dataset=${DATASET}"
echo "Node: \$SLURM_JOB_NODELIST   Start: \$(date)"

module load singularity/3.11.3
nvidia-smi -L || { echo "ERROR: no GPU visible"; exit 1; }

singularity exec --nv "${SIF}" \\
    python3 "${ABLATION_DIR}/${PY}" \\
        --dataset ${DATASET} \\
        --seed ${SEED} \\
        --folds 5 --fold ${FOLD} \\
        --loss ${LOSS} \\
        --tag-suffix _${NETWORK}_${LOSS} \\
        --optimizer adam_paperbc3d --lr 5e-4 \\
        ${EXTRA[*]:-}

echo "End: \$(date)"
EOF
    chmod +x "$OUT"
    echo "Wrote $OUT"
    if [[ "$DO_SUBMIT" -eq 1 ]]; then sbatch "$OUT"; fi
done

echo
echo "  network : $NETWORK -> $PY"
echo "  loss    : $LOSS   (baseline for comparison: dicefocal)"
echo "  folds   : $FOLDS   dataset: $DATASET   seed: $SEED"
echo "  tag     : ..._cv5f<F>_seed${SEED}_${NETWORK}_${LOSS}"
[[ "$DO_SUBMIT" -eq 0 ]] && echo && echo "Add --submit to launch."
echo
echo "When finished:  python3 compare_loss.py"
