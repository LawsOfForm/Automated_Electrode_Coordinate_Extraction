# MRI Ablation Study — Master Guide (Greifswald Brain HPC)

Single consolidated guide for running the 3D Attention U-Net ablation pipeline
on the Greifswald Brain cluster as `niemannf`. 

> **Cluster facts confirmed during setup (don't re-derive these):**
> - Container runtime: **`singularity/3.11.3`**.
> - GPU GRES label: **`gpu:A100:1`** — capital `A100`, case-sensitive. Lowercase
>   `a100` is rejected with "Requested node configuration is not available".
> - GPU nodes: `vision-[01-07]`, each with **4× A100-SXM4-80GB**, host driver
>   **595.71.05 (CUDA 13.2)**.
> - Working image: **`pytorch_monai.sif`** (rebuilt with the two fixes below).
> - Dataset root: **`~/Dataset/dataset_RU`** — the code walks 8 levels down to
>   the real subjects under `media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction/`.
>   72 paired (volume, mask) cases.

---

## Table of Contents

1. [Partitions & GPU selection](#1-partitions--gpu-selection)
2. [Build the container locally](#2-build-the-container-locally)
3. [Transfer to the cluster](#3-transfer-to-the-cluster-and-back)
4. [The dry-run testing ladder](#4-the-dry-run-testing-ladder)
5. [Known fixes baked into the image](#5-known-fixes-baked-into-the-image)
6. [Full CLI reference (all pipeline flags)](#6-full-cli-reference-all-pipeline-flags)
7. [Batch size & gradient accumulation](#7-batch-size--gradient-accumulation)
8. [Learning rate: too high vs too low](#8-learning-rate-too-high-vs-too-low)
   - [8.1 Adding LR warmup](#81-adding-lr-warmup-optional-lets-you-keep-a-higher-target-lr)
9. [The working SLURM script](#9-the-working-slurm-script)
10. [Monitoring, stopping & debugging jobs](#10-monitoring-stopping--debugging-jobs)
11. [Appendix: quick command reference](#11-appendix-quick-command-reference)
12. [Running several networks without overwriting each other](#12-running-several-networks-without-overwriting-each-other)

---

## 1. Partitions & GPU selection

Check GPU labels and availability:
```bash
sinfo -p vision -o "%P %N %G"        # -> vision vision-[01-07] gpu:A100:4
squeue -p vision                     # who's using the GPUs right now
```

| Partition    | Nodes            | GPUs            | Use case                    |
|:------------:|:----------------:|:---------------:|:----------------------------|
| **vision**   | vision-[01-07]   | **4× A100 80GB**| **Deep learning (use this)**|
| vision-fast  | vision-08        | A100            | Fast queue, 1 node          |
| pinky        | node[381-390]    | —               | Large-memory CPU            |
| snowball     | node[201-380]    | —               | CPU                         |
| batch        | node08x          | —               | CPU                         |
| storm        | storm-[01-05]    | —               | Small jobs                  |



---

## 2. Build the container locally

> **Never build the `.sif` on the cluster** — build locally, transfer the image.

```bash
cd 3D_local/Cluster
sudo singularity build pytorch_monai.sif pytorch_monai.def
ls -lh pytorch_monai.sif
```

The `.def` must contain the two fixes in [section 5](#5-known-fixes-baked-into-the-image).
If you skip them, the image will fail at runtime (NumPy crash, then CUDA Error 803).


```txt
ootstrap: docker
From: nvcr.io/nvidia/pytorch:24.10-py3
# This base image ships:
#   PyTorch 2.5, CUDA 12.6, cuDNN, NCCL, all pre-built for NVIDIA GPUs.
#   Switch to a different tag if the cluster's CUDA driver is older.

%post
    # Stay on the wheel index that ships with the base image so we don't
    # accidentally replace torch with a CPU build. MONAI is a thin layer
    # on top of torch, so we install it without dependency resolution.
    pip install --no-cache-dir --no-deps \
        monai==1.4.0
    # Everything else (these are pure-python or stable wheels):
    pip install --no-cache-dir \
        nibabel \
        tqdm \
        tensorboard \
        matplotlib \
        einops

    # IMPORTANT: the base image's torch (2.5/nv24.10) is compiled against
    # NumPy 1.x. Installing matplotlib/tensorboard above pulls in NumPy 2.x
    # as a transitive dependency, which breaks torch's C-extension ABI
    # ("_ARRAY_API not found"). Force NumPy back below 2.0 as the LAST pip
    # action so nothing can re-upgrade it. Confirmed working at numpy 1.26.4.
    pip install --no-cache-dir --force-reinstall "numpy<2"

    # Fail the build immediately if NumPy ever ends up >= 2.0 again, so a
    # silent dependency bump can't ship a broken image.
    python3 -c "import numpy, sys; v=numpy.__version__; print('numpy', v); sys.exit(0 if int(v.split('.')[0]) < 2 else 1)"

    # CUDA forward-compat layer removal.
    # The NGC base image ships /usr/local/cuda/compat/ (CUDA forward-
    # compatibility libs). On startup the NGC entrypoint tries to rm/relink
    # this dir, but a Singularity .sif is READ-ONLY at runtime, so the rm
    # fails and the stale compat libs shadow the host driver -> CUDA
    # "Error 803: unsupported display driver / cuda driver combination".
    # The cluster's host driver (595.71.05, CUDA 13.2) is NEWER than this
    # container's CUDA (12.6), so the compat layer is not needed at all --
    # we remove it here at BUILD time (while the fs is still writable) so
    # the container uses the host driver injected by --nv directly.
    rm -rf /usr/local/cuda/compat || true

%environment
    # MONAI emits a lot of deprecation warnings -- quiet them.
    export PYTHONWARNINGS="ignore::DeprecationWarning"
    # Keep PyTorch from grabbing more CPU threads than SLURM gave us.
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1

%labels
    Maintainer niemannf
    Purpose    Ablation study for the Automated Electrode Extraction paper
```
- safe this file as pytorch_monai.def
- creaet sif

```bash
singularity build pytorch_monai.def pytorch_monai.sif
```

---

## 3. Transfer to the cluster

### 3.1 Transfer manually 

```bash
scp -r folder/path/local/computer/ niemannf@brain.uni-greifswald.de:/folder/path/cluster
```

- from cluster to computer

```bash
scp -r niemannf@brain.uni-greifswald.de:/folder/path/cluster /media/folder/local/computer
```

### 3.2 Transfer per script

```bash
bash transfer_to_cluster.sh
```
Creates `~/ablation/{slurm,logs,ablation_results}` and rsyncs the `.py` scripts
and the `.sif`. Large folders (`debug_images/`, `models/`, `runs/`) are skipped.

> **Filename caveat:** the old transfer script references
> `c_model_channel_depth.py --increase-channels`, but the increased-channel
> model you actually run is **`c_increased_model_paper_bioarxiv.py`** (channels
> hardcoded to `(48, 96, 192, 384, 768)`, no `--increase-channels` flag). Make
> sure your SLURM script points at the file you intend.

---

## 4. The dry-run testing ladder

Run `dryrun.sh` on the **login node** first; it fails fast and locally so a
24-hour job doesn't die in minute 3. Work the rungs in order.

| Rung | Where        | Checks                          | Stop if            |
|-----:|--------------|---------------------------------|--------------------|
| 0    | local        | Python syntax (`py_compile`)    | traceback          |
| 1    | login        | files exist                     | any missing        |
| 2    | login        | log dir writable                | not writable       |
| 3    | login        | container runtime on PATH       | none found         |
| 4    | login        | dataset resolves (8-level walk) | no sub-* found     |
| 5    | login        | GRES label (case-sensitive)     | (record it)        |
| 6    | login        | imports inside container        | import error       |
| 7    | login        | `--help` parses                 | traceback          |
| 8    | GPU session  | CUDA True + end-to-end run      | CUDA False / crash |
| 9    | sbatch       | the real job                    | read `Reason=`     |

### Rung 0 — syntax (local)
```bash
python3 -m py_compile c_increased_model_paper_bioarxiv.py ablation_utils.py && echo OK
```

### Rungs 1–7 — login node
```bash
ssh niemannf@login-b
cd ~/ablation
bash dryrun.sh
```
`[FAIL]` notes:
- **Rung 4** must use an 8-level `find` (the subjects are 6 levels deep). A
  shallow search gives a *false* failure — the real `ablation_utils.py`
  resolves it. Confirm directly:
  ```bash
  singularity exec pytorch_monai_2.sif python3 -c "
  from ablation_utils import resolve_dataset_root, create_dataset
  print('root:', resolve_dataset_root('$HOME/Dataset/dataset_RU'))
  print('train cases:', len(create_dataset('train', validation_cases=2, test_cases=2, dataset_root='$HOME/Dataset/dataset_RU')))
  "
  ```
  Expect: root ending in `automated_electrode_extraction`, 72 cases, 68 train.
- **Rung 5** prints `gpu:A100:4` → use `--gres=gpu:A100:1` (capital).
- **Rung 6** must show torch + monai versions with **no NumPy ABI tracebacks**.
  The `No CUDA runtime` / `No NVIDIA driver` warnings on the login node are
  expected (no GPU there) — ignore them.

### Rung 8 — GPU smoke test (interactive)
Get a session with **enough time** (a 15-min limit expiring mid-test was a real
source of confusion):
```bash
srun --partition=vision --gres=gpu:A100:1 --mem=32G --cpus-per-task=4 --time=00:40:00 --pty bash
module load singularity/3.11.3
cd ~/ablation

nvidia-smi -L     # must list the A100s; if it errors you have no GPU — stop

# 8a. CUDA visible inside the container?
singularity exec --nv pytorch_monai_2.sif python3 -c \
  "import torch; print('avail:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expect: avail: True / NVIDIA A100-SXM4-80GB

# 8b. Tiny end-to-end run with the STABLE recipe (see section 8):
singularity exec --nv pytorch_monai_2.sif \
  python3 c_increased_model_paper_bioarxiv.py \
    --dataset-root ~/Dataset/dataset_RU \
    --optimizer adam_paperbc3d --lr 1e-4 \
    --max-iterations 200 \
    --num-workers 0 --no-pin-memory
```
Watch the tqdm `loss=` value: it must stay **finite** (no `nan`) past iter ~40.
At `--lr 1e-4` it sits around 0.37–0.42 and survives; at `1e-3` it diverges.

---

## 5. Known fixes baked into the image

Two problems were diagnosed and fixed in the `.def`. Both live in `%post`:

### Fix A — NumPy 1.x vs 2.x ABI crash
The NGC base image's torch is compiled against NumPy 1.x, but pip pulled in
NumPy 2.x as a transitive dependency (via matplotlib/tensorboard), giving
`_ARRAY_API not found` / "compiled using NumPy 1.x cannot be run in NumPy 2.2.6".
Fix — pin NumPy below 2.0 as the **last** pip action:
```bash
pip install --no-cache-dir --force-reinstall "numpy<2"
python3 -c "import numpy,sys; v=numpy.__version__; print('numpy',v); sys.exit(0 if int(v.split('.')[0])<2 else 1)"
```
Confirmed working at numpy 1.26.4.

### Fix B — CUDA compat layer / Error 803
NGC images ship `/usr/local/cuda/compat/`. On startup the entrypoint tries to
`rm` it, but a `.sif` is **read-only**, so the stale compat libs shadow the host
driver → `Error 803: unsupported display driver / cuda driver combination`. The
host driver (595 / CUDA 13.2) is **newer** than the container's CUDA 12.6, so the
compat layer isn't needed — remove it at build time:
```bash
rm -rf /usr/local/cuda/compat || true
```

> The `15:4: not a valid test operator` lines you see on every run are harmless
> noise from an NGC env script reacting to a host variable. Add `--cleanenv` to
> the `singularity exec` to silence them; they do not affect the run.

---

## 6. Full CLI reference (all pipeline flags)

Every flag below is from `build_common_argparser` in `ablation_utils.py`.
Run `... python3 c_increased_model_paper_bioarxiv.py --help` to see them live.

### Core training
| Flag | Default | Meaning |
|---|---|---|
| `--dataset-root PATH` | env/candidates | Dataset root; 8-level walk finds subjects |
| `--max-iterations N` | 50000 | Total training iterations |
| `--eval-num N` | 500 | Validate every N iters (also checkpoint cadence) |
| `--seed N` | 1001 | RNG seed |
| `--validation-cases N` | 8 | Held-out validation subjects |
| `--test-cases N` | 8 | Held-out test subjects |

### Batch / throughput
| Flag | Default | Meaning |
|---|---|---|
| `--batch-size N` | 1 | Real samples per forward pass (GPU-memory bound) |
| `--accumulation-steps N` | 8 | Micro-batches accumulated before a weight update |
| `--num-workers N` | 2 | DataLoader workers. Use `0` if you hit FD errors |
| `--no-pin-memory` | off | Disable pin_memory (FD-exhaustion workaround) |
| `--prefetch-factor N` | 2 | Batches prefetched per worker |

> **Effective batch = `--batch-size` × `--accumulation-steps`.**
> Default is 1 × 8 = **8**. See [section 7](#7-batch-size--gradient-accumulation).

### Optimizer & loss
| Flag | Choices / Default | Meaning |
|---|---|---|
| `--optimizer` | `paper` (def), `adamax`, `adam_paperbc3d` | Optimizer recipe — see below |
| `--lr FLOAT` | 0.005 | Learning rate. **IGNORED when `--optimizer paper`** |
| `--loss` | `dicefocal` (def), `diceloss` | Loss function |

**Optimizer recipes (exact):**
- `paper` → `Adam(net.parameters())` — hardcoded lr=1e-3, wd=0. **Ignores `--lr`.**
- `adamax` → `Adamax(lr=<--lr>, betas=(0.95,0.99), wd=1e-5)`. Respects `--lr`.
- `adam_paperbc3d` → `Adam(lr=<--lr>, weight_decay=1e-4)`. Respects `--lr`.

**Loss recipes:**
- `dicefocal` → `DiceFocalLoss(lambda_dice=0.3, lambda_focal=0.7, gamma=2.5, include_background=False)` — what the a/b/c/d ablations use. **Keep this for comparability.**
- `diceloss` → `DiceLoss(to_onehot_y=True, softmax=True)` — the model_bc_3d recipe.

### Sanity-check / early-stop knobs
| Flag | Default | Meaning |
|---|---|---|
| `--min-dice-threshold FLOAT` | 0.05 | Val Dice the model must exceed once to count as "learning" |
| `--sanity-probation-iters N` | 15000 | No eval-based abort before this iter |
| `--max-no-improve-evals N` | 40 | Abort if still below threshold AND no improvement this many evals |

> **Two distinct sanity checks** (this is what aborted the early runs):
> 1. **Loss check** — aborts on **5 consecutive non-finite (NaN/Inf) losses**,
>    active from iter 0. Isolated NaN spikes are tolerated (the counter
>    decrements on good steps); only sustained divergence aborts.
> 2. **Eval check** — only fires *after* iter 15000, and only if Dice never once
>    crossed 0.05. Once Dice crosses 0.05, it's permanently disabled.

---

## 7. Batch size & gradient accumulation

You can raise the batch two ways:

**A) Real batch size** (`--batch-size`): more samples per forward pass.
Increases GPU memory use. With 80 GB A100s and 3D PETRA volumes you may fit a
small real batch, but 3D volumes are memory-hungry — increase cautiously and
watch `nvidia-smi` memory.
```bash
--batch-size 2 --accumulation-steps 4    # effective batch still 8, fewer accum steps
```

**B) Gradient accumulation** (`--accumulation-steps`): accumulate gradients over
several micro-batches before stepping. Gives a large *effective* batch with
batch_size=1 memory cost — which is why the default is `1 × 8 = 8`.

**Effective batch = batch_size × accumulation_steps.** To keep results
comparable, hold the *effective* batch constant. E.g. these are equivalent in
effective batch but differ in speed/memory:
| `--batch-size` | `--accumulation-steps` | Effective | Memory | Speed |
|---:|---:|---:|---|---|
| 1 | 8 | 8 | low | slower (8 fwd/step) |
| 2 | 4 | 8 | higher | faster |
| 4 | 2 | 8 | highest | fastest |

> **Does increasing batch size help a bad learning rate?** Largely **no** —
> see the next section. Batch size is a throughput/memory knob, not a divergence
> fix. If you raise the *real* batch a lot, the linear-scaling heuristic says
> you'd *raise* LR to match — the opposite of what fixed your divergence.

---

## 8. Learning rate: too high vs too low

This is the issue that aborted the first real runs. **Diagnosis: LR too high.**

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss → `nan`/`inf` in first tens–hundreds of iters; "5 consecutive non-finite loss" abort | **LR too high** (divergence) | **Lower LR.** 1e-3 diverged here; **1e-4 is stable** |
| Loss falls then explodes later | LR slightly high / unlucky batch | Lower LR, or add warmup |
| Loss finite but Dice stuck near 0 for a very long time; very slow progress | LR possibly **too low**, or genuinely hard init | Raise LR moderately (e.g. 3e-4, 5e-4); check data/labels |
| Loss decreases steadily, Dice climbs | LR about right | Leave it |

**What was observed on this exact model (53M-param increased-channel net):**
- `--optimizer paper` (lr=1e-3, dicefocal) → **diverged at iter 37.**
- `--optimizer adam_paperbc3d --lr 1e-3 --loss diceloss` → **diverged at iter 27.**
  (So the loss swap did *not* fix it — the LR did.)
- `--optimizer adam_paperbc3d --lr 1e-4` (dicefocal) → **stable**, loss ~0.37,
  ran past iter 98+ with no NaN. ✅ This is the recipe to use.

**Can batch size help instead of changing LR?** Not for divergence. A bigger
batch smooths gradient *noise*, but your NaN came from the *magnitude* of the
update (LR × gradient) at a hot LR, not from noise — and the effective batch was
already 8 via accumulation. If anything, a larger batch invites a *higher* LR
(linear scaling rule), which is the wrong direction here. **Lowering the LR is
the fix; batch size is for speed/memory.**

> ⚠ **Important caveats on the LR fix:**
> - 200 iters proves it *doesn't diverge*; it does **not** prove it *learns
>   well*. Val Dice near 0 that early is normal. Only a long run shows final
>   quality. 1e-4 could even be *too low* to converge well within 50k steps — if
>   Dice plateaus, try 3e-4 / 5e-4 or add warmup.
> - **Comparability:** if the other variants (a, b, c_reduced) trained stably at
>   the `paper` recipe, running *this* one at `adam_paperbc3d`/1e-4 makes it not
>   strictly apples-to-apples. The bigger net genuinely needs the lower LR — a
>   defensible reason — but decide whether to note it or re-run the set under one
>   recipe. This is a **study-design** decision, not just engineering.

### 8.1 Adding LR warmup (optional, lets you keep a higher target LR)

Warmup is often cleaner than just dropping the LR: it ramps the LR up gently
over the first few hundred steps (when the randomly-initialised net is most
unstable), then runs your full target LR — giving better final convergence than
a permanently low LR while still avoiding the iter-37 explosion.

**There is no `--warmup` flag** — the argparser doesn't expose one, so this is a
**source edit** to `ablation_utils.py`. (The `.py` files live in `~/ablation/`
outside the `.sif`, so editing them takes effect immediately — **no rebuild
needed.**)

**The edit.** Replace the single scheduler line (originally:
`self.scheduler = CosineAnnealingWarmRestarts(self.opt, T_0=1000, T_mult=1, eta_min=1e-5)`)
with a `LinearLR` warmup chained into the same cosine schedule via
`SequentialLR`:

```python
# --- LR schedule: linear warmup -> cosine-annealing-with-restarts ---
# NB: scheduler.step() is called once per OPTIMIZER step (every
# accumulation_steps micro-batches), NOT every micro-batch. So warmup_steps
# is counted in OPTIMIZER steps.
warmup_steps = 200          # optimizer steps of warmup; tune as needed
warmup = torch.optim.lr_scheduler.LinearLR(
    self.opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
)
cosine = CosineAnnealingWarmRestarts(self.opt, T_0=1000, T_mult=1, eta_min=1e-5)
self.scheduler = torch.optim.lr_scheduler.SequentialLR(
    self.opt, schedulers=[warmup, cosine], milestones=[warmup_steps]
)
```

What it does: for the first 200 optimizer steps the LR rises linearly from 1% of
the target (`start_factor=0.01`) to the full target, then `SequentialLR` hands
off to the original cosine-with-restarts schedule unchanged.

**Using it with a higher target LR** — warmup's whole point is that you can now
go back *up* from the conservative 1e-4:
```bash
singularity exec --nv ~/ablation/pytorch_monai_2.sif \
  python3 ~/ablation/c_increased_model_paper_bioarxiv.py \
    --dataset-root ~/Dataset/dataset_RU \
    --optimizer adam_paperbc3d --lr 5e-4 \
    --max-iterations 300 --num-workers 0 --no-pin-memory
```
Watch the tqdm `loss=` (stays finite through warmup?) and the TensorBoard `LR`
scalar (climbs over the first 200 steps, then follows the cosine sawtooth).

**Tuning:** `warmup_steps=200` is a starting guess chosen to cover the iter-27–37
danger zone — lengthen to 500 for a higher target LR, shorten if 200 is plenty.

> ⚠ **Caveats specific to warmup:**
> - **Verify the handoff against your torch version (2.5.0a0+nv24.10).**
>   `SequentialLR` chained with `CosineAnnealingWarmRestarts` has had edge cases
>   across releases — `WarmRestarts` keeps its own internal step counter. After
>   warmup, the logged `LR` scalar should jump onto the cosine curve and restart
>   every 1000 steps. If it looks flat or wrong, fall back to plain
>   `CosineAnnealingLR` for the tail, or do warmup as a manual LR override for
>   the first N steps instead of via `SequentialLR`. **Confirm from the LR
>   curve — don't assume it's clean.**
> - The "5 consecutive NaN" loss check is unchanged: if the target LR is so high
>   that even the warmup tail diverges, it still aborts (that's correct).
> - Warmup changes this variant's recipe vs the others — same comparability note
>   as above.

---

## 9. The working SLURM script

`~/ablation/slurm/slurm_c_increased.sh` — adapted from the cluster's MPI
template to a single-node, single-GPU, containerized run, with the stable LR
recipe and all fixes:

```bash
#!/bin/bash
#SBATCH -J ablation_c_increased
#SBATCH --partition=vision
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1                       # capital A100 — case-sensitive
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o /home/niemannf/ablation/logs/c_increased_%j.out   # absolute path (no ~)
#SBATCH -e /home/niemannf/ablation/logs/c_increased_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niemannf@uni-greifswald.de  # <-- verify this address

set -euo pipefail

echo "Job $SLURM_JOB_ID on $SLURM_JOB_NODELIST  GPUs=${CUDA_VISIBLE_DEVICES:-none}"
echo "Start: $(date)"

module load singularity/3.11.3

ABLATION_DIR="/home/niemannf/ablation"
SIF="$ABLATION_DIR/pytorch_monai.sif"
PY_SCRIPT="$ABLATION_DIR/c_increased_model_paper_bioarxiv.py"
DATASET_ROOT="/home/niemannf/Dataset/dataset_RU"

nvidia-smi -L || { echo "ERROR: no GPU visible"; exit 1; }

singularity exec --nv "$SIF" \
    python3 "$PY_SCRIPT" \
        --dataset-root "$DATASET_ROOT" \
        --optimizer adam_paperbc3d --lr 1e-4
        # full 50,000 iters (script default). For a real batch change add e.g.
        # --batch-size 2 --accumulation-steps 4   (keeps effective batch = 8)

echo "End: $(date)"
```

Submit and watch:
```bash
sbatch ~/ablation/slurm/slurm_c_increased.sh
```

> Drop the smoke-test flags (`--num-workers 0 --no-pin-memory --max-iterations 200`)
> for the production run so the DataLoader runs at full speed and the full
> schedule executes. Use `sbatch` (not an interactive `srun`) for the long run so
> it survives SSH disconnects and isn't killed by an `srun --time` limit.

---

## 10. Monitoring, stopping & debugging jobs

**Is it running / pending?**
```bash
squeue -u niemannf
squeue -j <jobid>              # "Invalid job id" = already finished/left queue
squeue -j <jobid> --start     # estimated start if PENDING
scontrol show job <jobid>     # full detail; read Reason= and JobState
```

**Live output** (training/tqdm goes to **stderr**):
```bash
tail -f ~/ablation/logs/c_increased_<jobid>.err
tail -f ~/ablation/logs/c_increased_<jobid>.out
ls -lt ~/ablation/logs/ | head    # if unsure of the filename
```

**After it finishes — success or failure?** (`scontrol` forgets ended jobs):
```bash
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,Start,End
```
`State`: `COMPLETED` / `FAILED` / `TIMEOUT` / `OUT_OF_MEMORY`. `ExitCode 0:0` = clean.
A `COMPLETED` job can still be a *training* failure — check the report:
```bash
cat ~/ablation/ablation_results/paper_attention_unet_report.md
```
A divergence shows "⚠ TRAINING FAILED" and "Loss diverged" at the top.

**GPU activity** (from another shell on the same node):
```bash
watch -n 2 nvidia-smi         # GPU-Util ~80-100% = training; 0% = stalled/eval
```

**Stop a job:**
```bash
scancel <jobid>               # cancel one batch job (stop = delete; no separate delete)
scancel -u niemannf           # cancel ALL your jobs (careful)
scancel --state=PENDING -u niemannf   # cancel only queued jobs
```
For an **interactive** foreground run: `Ctrl-C` stops the Python process;
`exit` releases the `srun` allocation.

**Common PENDING reasons:**
- `(Resources)` / `(Priority)` → normal queueing, wait.
- `(ReqNodeNotAvail)` → often a bad/mismatched GRES label (use capital `A100`).
- `(QOSMaxJobsPerUserLimit)` → at a scheduler limit.

---

## 11. Appendix: quick command reference

```bash
# Environment
module load singularity/3.11.3
sinfo -p vision -o "%P %N %G"                 # GPU labels
squeue -u niemannf                            # my jobs

# Interactive GPU session (generous time limit)
srun --partition=vision --gres=gpu:A100:1 --mem=32G --cpus-per-task=4 --time=00:40:00 --pty bash

# Stable smoke test (200 iters)
singularity exec --nv ~/ablation/pytorch_monai_2.sif \
  python3 ~/ablation/c_increased_model_paper_bioarxiv.py \
    --dataset-root ~/Dataset/dataset_RU \
    --optimizer adam_paperbc3d --lr 1e-4 \
    --max-iterations 200 --num-workers 0 --no-pin-memory

# Full run
sbatch ~/ablation/slurm/slurm_c_increased.sh

# Monitor / stop
tail -f ~/ablation/logs/c_increased_<jobid>.err
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed
scancel <jobid>
```

### Stable recipe (memorize this one)
```
--optimizer adam_paperbc3d --lr 1e-4        # dicefocal loss (default), effective batch 8
```
`paper` recipe (lr=1e-3) diverges on the 53M-param increased-channel net.
Lower the LR for stability; batch size is for speed/memory, not divergence.

---

## 12. Running several networks without overwriting each other

### The problem
Checkpoints and reports are named after the script's `experiment_tag`:
`ablation_utils.py` builds `f"{experiment_tag}_best_metric_model.pth"` and
`f"{experiment_tag}_report.md/.txt/.json"`, plus a TensorBoard dir named after
the tag. That namespacing only helps if the tag is **unique per run** — and it
isn't: both `c_normal_model_paper_bioarxiv.py` and
`c_increased_model_paper_bioarxiv.py` **hardcode the same**
`EXPERIMENT_TAG = "paper_attention_unet"`. Run them together and they clobber
each other's checkpoint, all three report files, and the TensorBoard run dir.
(`d_model_segresnet.py` is safe — its tag includes `init_filters`, e.g.
`d_segresnet_f16`.)

### The fix: `--tag-suffix` (one flag, no edits to the model files)
The shared argparser now has a `--tag-suffix` flag that is appended to the tag
inside `run_experiment_from_args`. It defaults to `""` (no change), so existing
behaviour is untouched. A leading underscore is added automatically.

```bash
# These now write to DIFFERENT files instead of colliding:
python3 c_normal_model_paper_bioarxiv.py    ... --tag-suffix _normal
python3 c_increased_model_paper_bioarxiv.py ... --tag-suffix _increased
# -> paper_attention_unet_normal_*    and    paper_attention_unet_increased_*
```

It also disambiguates **seeds** of the same model: `--tag-suffix _seed2002`.

> This requires the patched `ablation_utils.py` (the one with `--tag-suffix`).
> On an unpatched copy the flag errors as "unrecognized arguments". The `.py`
> files live outside the `.sif`, so updating `ablation_utils.py` needs no
> rebuild.

### The helper: `make_slurm.sh` (generate a correct submit script per network)
Rather than hand-editing the script name, job name, log paths and suffix each
time, `make_slurm.sh` generates a ready-to-submit SLURM file for one network
with a unique `--tag-suffix` already wired in. It writes
`slurm/slurm_<label>.sh` and prints the `sbatch` command — it does **not**
submit, so you can review first.

```bash
./make_slurm.sh c_increased
./make_slurm.sh c_normal
./make_slurm.sh segresnet --init-filters 16 --auto-tune-batch
./make_slurm.sh c_increased --seed 2002 --tag-suffix _increased_seed2002
```

Recognised network names and the script / suffix each maps to:

| Name             | Python script                          | Auto suffix     |
|------------------|----------------------------------------|-----------------|
| `c_normal`       | `c_normal_model_paper_bioarxiv.py`     | `_normal`       |
| `c_increased`    | `c_increased_model_paper_bioarxiv.py`  | `_increased`    |
| `segresnet`      | `d_model_segresnet.py`                  | (tag has `init_filters`) |
| `a_baseline`     | `a_model_unet_baseline.py`              | `_a_baseline`   |
| `b_no_attention` | `b_model_proposed_no_attention.py`      | `_b_no_attn`    |

Behaviour notes:
- Any extra args after the network name are passed straight to the Python
  script (`--seed`, `--init-filters`, `--max-iterations`, etc.).
- If you pass your own `--tag-suffix`, the generator uses it (no double suffix)
  and names the file after it, so each seed/variant gets a unique filename.
- Defaults baked in (override via env vars or extra args): `ABLATION_DIR`,
  `SIF=pytorch_monai_2.sif`, `DATASET_ROOT`, `MAIL_USER`, partition `vision`,
  `--gres=gpu:A100:1`, and the stable recipe `--optimizer adam_paperbc3d
  --lr 5e-4`. Example override: `DATASET_ROOT=/path ./make_slurm.sh c_normal`.

Run all three at once, no collision:
```bash
./make_slurm.sh c_normal     && sbatch slurm/slurm_normal.sh
./make_slurm.sh c_increased  && sbatch slurm/slurm_increased.sh
./make_slurm.sh segresnet --init-filters 16 && sbatch slurm/slurm_segresnet.sh
```

> **Comparability reminder:** the generator applies the same
> `adam_paperbc3d --lr 5e-4` recipe to every network. That keeps the ablation
> apples-to-apples *only if* that recipe is stable for each architecture — it
> was verified on the AttentionUnet variants, not on SegResNet or the a/b
> baselines. Smoke-test a new architecture briefly before committing a 72h job.
