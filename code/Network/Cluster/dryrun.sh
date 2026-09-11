#!/usr/bin/env bash
# dryrun.sh — verify the environment BEFORE submitting to SLURM.
# Run on the login node first, then (rung 7) inside an interactive GPU session.
# Nothing here calls sbatch. It only checks that each piece exists/works.

set -uo pipefail   # NOT -e: we want to keep going and report every failure

# ---- EDIT THESE TWO IF NEEDED ------------------------------------------- #
ABLATION_DIR="$HOME/ablation"
SIF="$ABLATION_DIR/pytorch_monai_2.sif"
# Script you actually want to run (you uploaded c_increased_model_paper_bioarxiv.py):
PY_SCRIPT="$ABLATION_DIR/c_increased_model_paper_bioarxiv.py"
# Candidate dataset roots — first existing one wins. Add/adjust as needed.
DATASET_CANDIDATES=(
    "/media/data04/Automatic_Electrode_extraction/Dataset/dataset_RU"
    "$HOME/Dataset/dataset_RU"
)
# ------------------------------------------------------------------------- #

pass() { echo -e "  \033[1;32m[PASS]\033[0m $1"; }
fail() { echo -e "  \033[1;31m[FAIL]\033[0m $1"; }
info() { echo -e "\033[1;34m== $1 ==\033[0m"; }

FAILED=0

info "1. Files present"
for f in "$SIF" "$PY_SCRIPT" "$ABLATION_DIR/ablation_utils.py"; do
    if [[ -f "$f" ]]; then pass "exists: $f"; else fail "MISSING: $f"; FAILED=1; fi
done

info "2. Log dir writable (SLURM won't expand ~ in #SBATCH — use absolute paths)"
mkdir -p "$ABLATION_DIR/logs" && [[ -w "$ABLATION_DIR/logs" ]] \
    && pass "writable: $ABLATION_DIR/logs" \
    || { fail "cannot write logs dir"; FAILED=1; }

info "3. Singularity / Apptainer module"
if command -v singularity >/dev/null 2>&1; then
    pass "singularity on PATH: $(command -v singularity)"
elif command -v apptainer >/dev/null 2>&1; then
    pass "apptainer on PATH: $(command -v apptainer) (use 'apptainer' instead of 'singularity')"
else
    fail "neither singularity nor apptainer on PATH — run 'module avail' and load the right one"
    echo "    Available modules matching 'sing/apptainer':"
    module avail 2>&1 | grep -iE "sing|apptainer" || echo "    (none found via module avail)"
    FAILED=1
fi
CONTAINER_CMD="$(command -v singularity || command -v apptainer || echo singularity)"

info "4. Resolve dataset root"
DATASET_ROOT=""
for d in "${DATASET_CANDIDATES[@]}"; do
    if [[ -d "$d" ]]; then DATASET_ROOT="$d"; pass "found dataset: $d"; break; fi
done
if [[ -z "$DATASET_ROOT" ]]; then
    fail "no dataset candidate exists. Tried: ${DATASET_CANDIDATES[*]}"
    FAILED=1
else
    n_sub=$(find "$DATASET_ROOT" -maxdepth 3 -type d -name 'sub-*' 2>/dev/null | wc -l)
    [[ "$n_sub" -gt 0 ]] && pass "found $n_sub sub-* folder(s)" \
        || fail "no sub-* subject folders under $DATASET_ROOT (resolve_dataset_root will raise)"
fi

info "5. GPU GRES label as SLURM actually reports it (case-sensitive!)"
echo "  vision partition GRES:"
sinfo -p vision -o "%P %N %G" 2>/dev/null | sed 's/^/    /' || echo "    (sinfo failed)"
echo "  -> use the EXACT label shown above in --gres=gpu:<label>:1"

info "6. Container can import torch + monai and see CUDA build (CPU login node OK)"
if [[ -f "$SIF" ]]; then
    "$CONTAINER_CMD" exec "$SIF" python3 -c \
"import torch, monai; print('torch', torch.__version__, '| monai', monai.__version__, '| cuda compiled:', torch.version.cuda)" \
        2>&1 | sed 's/^/    /' && pass "imports OK inside container" \
        || { fail "import failed inside container"; FAILED=1; }
fi

info "7. Script --help parses (catches argparse/import errors fast)"
if [[ -f "$SIF" && -f "$PY_SCRIPT" ]]; then
    "$CONTAINER_CMD" exec "$SIF" python3 "$PY_SCRIPT" --help >/dev/null 2>/tmp/help_err \
        && pass "--help works (argparse + top-level imports fine)" \
        || { fail "--help failed:"; sed 's/^/      /' /tmp/help_err; FAILED=1; }
fi

echo
if [[ "$FAILED" -eq 0 ]]; then
    echo -e "\033[1;32mALL CHECKS PASSED on login node.\033[0m"
    echo "Next: rung 8 (GPU smoke test) inside an interactive session — see notes."
    [[ -n "$DATASET_ROOT" ]] && echo "Resolved DATASET_ROOT=$DATASET_ROOT"
else
    echo -e "\033[1;31mSome checks FAILED — fix the [FAIL] lines above before sbatch.\033[0m"
fi
