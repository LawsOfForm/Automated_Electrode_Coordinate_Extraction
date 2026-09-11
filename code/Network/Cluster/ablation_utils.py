"""
Shared utilities for the ablation study scripts (a_*, b_*, c_*).

This module contains everything that must stay IDENTICAL across the ablation
variants so that observed differences are attributable only to the model
architecture, not to data, training, or evaluation differences:

    - dataset construction and transforms (identical to the original paper)
    - the `Network` training/validation/test loop
    - a unified results writer that produces a per-experiment Markdown report
      and a machine-readable TXT summary

The ONLY thing the individual a_/b_/c_ scripts customise is the model
(`build_model()`-style factory passed into `run_experiment`) and an
experiment tag used for output filenames and the TensorBoard log dir.
"""

from __future__ import annotations

import csv
import json
import os
import os.path as op
import platform
from datetime import datetime
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import monai.transforms as tfms
import numpy as np
import torch
from monai.data import ArrayDataset, DataLoader, decollate_batch
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Subject-wise splitting + grouped k-fold CV. Keep subject_wise_split.py in the
# same directory. See CHANGELOG_split_fix.md for why the image-wise split leaked.
from subject_wise_split import (
    subject_wise_subsetting,
    subsetting_legacy_imagewise,
    kfold_subsetting,
    final_subsetting,
    extract_subject_id,
)


# --------------------------------------------------------------------------- #
# Multiprocessing / file-descriptor robustness
# --------------------------------------------------------------------------- #
# The original loader configuration (num_workers=4, pin_memory=True) blows
# through the default Linux soft FD limit (1024) for big 3D volumes, which
# manifests as
#     RuntimeError: received 0 items of ancdata
#     RuntimeError: Pin memory thread exited unexpectedly
# To make the loader robust we:
#   1. switch torch.multiprocessing to the 'file_system' sharing strategy
#      (uses /dev/shm named segments instead of passing per-tensor FDs
#      through Unix sockets -- the FD count no longer scales with the
#      number of worker-to-main tensor transfers),
#   2. raise the soft FD limit of the current process up to the hard
#      limit (usually 4096 or 1048576 on modern kernels),
#   3. configure DataLoaders with persistent_workers=True so workers are
#      not torn down + recreated each epoch (which leaks FDs over time).
def _configure_multiprocessing_for_large_volumes():
    """Apply FD / sharing-strategy fixes for 3D-volume DataLoaders.

    Safe to call multiple times. Should run before any DataLoader is
    constructed.
    """
    # 1) file_system sharing strategy
    try:
        import torch.multiprocessing as mp
        if mp.get_sharing_strategy() != "file_system":
            mp.set_sharing_strategy("file_system")
    except Exception as e:
        print(f"[mp-setup] could not switch sharing strategy: {e}")

    # 2) raise soft FD limit
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard if hard != resource.RLIM_INFINITY else 65536
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            print(f"[mp-setup] raised RLIMIT_NOFILE from {soft} to {target}")
    except Exception as e:
        # Non-Unix or insufficient privileges -- not fatal.
        print(f"[mp-setup] could not raise RLIMIT_NOFILE: {e}")


_configure_multiprocessing_for_large_volumes()

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).parent.resolve()
DEBUG_DIR_DEFAULT = SCRIPT_DIR / "debug_images"
RESULTS_DIR_DEFAULT = SCRIPT_DIR / "ablation_results"

# Filled in by create_dataset(); used to verify no leakage and to write a
# per-run manifest recording exactly which participant went to which split.
SPLIT_MANIFEST: dict = {}
RUNS_DIR_DEFAULT = SCRIPT_DIR / "runs"

# Dataset root resolution.
#
# The original paper hard-coded a single path on /media/data01, but the
# T1w-only dataset the reviewer is being re-run against lives on /media/data04
# under a different folder name (`dataset_T1`). To avoid this happening again
# we try, in order:
#
#   1. the env var `ELECTRODE_DATASET_ROOT` (overrides everything),
#   2. the candidates listed in `CANDIDATE_DATASET_ROOTS` (first existing wins).
#
# Each "root" is the path *down to* `.../automated_electrode_extraction/`,
# i.e. the directory that directly contains the `sub-*` subject folders.

CANDIDATE_DATASET_DIRS = [
    # Each entry is a "dataset folder" — the code will automatically find
    # whichever subdirectory inside it contains the sub-* subject folders.
    # That way we never need to hardcode the internal nesting again.
    "/media/data04/Automatic_Electrode_extraction/Dataset/dataset_RU"
    #"/media/data04/Automatic_Electrode_extraction/Dataset/dataset_T1",
    #"/media/data01/Automatic_Electrode_extraction/Dataset/dataset",
    #"/media/data01/Automatic_Electrode_extraction/Dataset/dataset_cut",
]


def _find_subject_root(top_dir):
    """Walk *top_dir* and return the directory that directly contains sub-*
    subject folders, or None if none is found.

    Handles every known nesting pattern:
        top_dir/sub-001/...                       (flat)
        top_dir/media/MeMoSLAP_Subjects/.../sub-001/...  (deep)

    Walks at most 8 levels deep to avoid searching the whole filesystem.
    """
    MAX_DEPTH = 8
    for dirpath, dirnames, _ in os.walk(top_dir):
        depth = dirpath.replace(top_dir, "").count(os.sep)
        if depth >= MAX_DEPTH:
            dirnames.clear()          # prune os.walk
            continue
        if any(d.startswith("sub-") for d in dirnames):
            return dirpath
    return None


def resolve_dataset_root(explicit=None):
    """Pick a dataset root that actually exists on disk.

    The "root" is the directory that *directly contains* the ``sub-*``
    subject folders (e.g. ``.../automated_electrode_extraction/``).

    Resolution order:
      1. ``explicit`` argument (--dataset-root CLI flag): may be the
         deep path already, or a top-level dataset folder — we try both.
      2. ``$ELECTRODE_DATASET_ROOT`` env var — same logic.
      3. First match in ``CANDIDATE_DATASET_DIRS`` (auto-discovers the
         sub-* root inside each folder).

    Raises FileNotFoundError with a diagnostic if nothing works.
    """
    def _resolve_one(path):
        """Given a user-supplied or candidate path, return the sub-* root."""
        if not op.isdir(path):
            return None
        # Maybe the path itself already contains sub-* folders?
        entries = os.listdir(path)
        if any(e.startswith("sub-") for e in entries):
            return path
        # Otherwise search inside
        return _find_subject_root(path)

    # 1. Explicit argument
    if explicit:
        found = _resolve_one(explicit)
        if found:
            return found
        raise FileNotFoundError(
            f"No sub-* subject folders found under the explicitly-"
            f"requested path: {explicit}"
        )

    # 2. Environment variable
    env_root = os.environ.get("ELECTRODE_DATASET_ROOT")
    if env_root:
        found = _resolve_one(env_root)
        if found:
            return found
        raise FileNotFoundError(
            f"$ELECTRODE_DATASET_ROOT is set to '{env_root}' but no sub-* "
            f"subject folders were found underneath it."
        )

    # 3. Built-in candidate list
    for cand in CANDIDATE_DATASET_DIRS:
        found = _resolve_one(cand)
        if found:
            return found

    tried = "\n  ".join(CANDIDATE_DATASET_DIRS)
    raise FileNotFoundError(
        "No dataset root found. Tried:\n  "
        + tried
        + "\nSet the ELECTRODE_DATASET_ROOT environment variable to override, "
          "or pass --dataset-root on the command line."
    )


# --------------------------------------------------------------------------- #
# Debug image saving (kept from the original paper code)
# --------------------------------------------------------------------------- #
def save_debug_batch(inputs, labels, predictions=None, debug_dir=DEBUG_DIR_DEFAULT, tag=""):
    """Save a slice with mask voxels for qualitative inspection during training."""
    os.makedirs(debug_dir, exist_ok=True)
    inputs = inputs.cpu().numpy()
    labels = labels.cpu().numpy()
    if predictions is not None:
        predictions = predictions.cpu().numpy()

    for i in range(min(3, inputs.shape[0])):
        mask_slices = np.sum(labels[i, 0], axis=(0, 1)) > 0
        if not np.any(mask_slices):
            continue

        best_slice_idx = int(np.argmax(np.sum(labels[i, 0], axis=(0, 1))))

        fig, axes = plt.subplots(1, 3 if predictions is not None else 2, figsize=(15, 5))
        axes[0].imshow(inputs[i, 0, :, :, best_slice_idx], cmap="gray")
        axes[0].set_title(f"Input (slice {best_slice_idx})")
        axes[0].axis("off")
        axes[1].imshow(labels[i, 0, :, :, best_slice_idx], cmap="jet", alpha=0.5)
        axes[1].set_title(f"GT (slice {best_slice_idx})")
        axes[1].axis("off")
        if predictions is not None:
            axes[2].imshow(predictions[i, 0, :, :, best_slice_idx], cmap="jet", alpha=0.5)
            axes[2].set_title(f"Pred (slice {best_slice_idx})")
            axes[2].axis("off")

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_{tag}_{timestamp}_sample_{i}_slice_{best_slice_idx}.png"
        plt.savefig(op.join(debug_dir, filename))
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Dataset construction (verbatim logic from the original paper)
# --------------------------------------------------------------------------- #
def subsetting(subset, vols, mask, validation_cases, test_cases, seed=42,
               split_mode="subject", folds=0, fold=0, verbose=True):
    """Deterministic train/val/test split.

    split_mode:
      "subject"      DEFAULT. Whole participants go to exactly one split, so no
                     participant appears in more than one of train/val/test.
                     Required: participants contribute up to 4 images each
                     (sessions x runs), and those are near-duplicate scans of
                     the same head under the same montage.
      "legacy_image" Original image-wise shuffle. Reproduces published results
                     only; NOT leakage-free.

    If ``folds`` >= 2 the k-fold cross-validation split is used instead
    (subject-wise by construction) and validation_cases/test_cases are ignored.
    """
    if split_mode == "final":
        return final_subsetting(subset, vols, mask, validation_cases,
                                seed=seed, verbose=verbose)
    if folds and folds >= 2:
        return kfold_subsetting(subset, vols, mask, fold=fold, n_folds=folds,
                                seed=seed, verbose=verbose)
    if split_mode == "subject":
        return subject_wise_subsetting(subset, vols, mask, validation_cases,
                                       test_cases, seed=seed, verbose=verbose)
    if split_mode == "legacy_image":
        return subsetting_legacy_imagewise(subset, vols, mask, validation_cases,
                                           test_cases, seed=seed)
    raise ValueError(f"unknown split_mode {split_mode!r}")


def _extract_join_key(filename_or_path):
    """Extract a deterministic join key from a volume filename or mask folder name.

    The key used to pair volumes with masks must be derived from the parts
    that are *shared* between the two naming conventions.  This varies across
    dataset variants:

    dataset_RU  (current)
      Volume:      rsub-001_ses-3_acq-petra_run-01_PDw.nii
      Mask folder: sub-001_ses-3_run-01
      → sub, ses, AND run are all present in both → key = "sub-001_ses-3_run-01"

    dataset_T1  (previous)
      Volume:      rsub-001_ses-3_acq-mprage_T1w.nii   (no run!)
      Mask folder: sub-001_ses-3_run-01
      → only sub and ses are shared → key = "sub-001_ses-3"

    This function detects which fields are present and builds the most
    specific key possible.  For volumes from dataset_T1 the run is absent,
    so we fall back to sub+ses and _pair_volumes_and_masks handles the
    remaining ambiguity by keeping the first volume per key.

    Returns None if sub or ses cannot be extracted.
    """
    import re
    sub = re.search(r"sub-(\d+)", filename_or_path)
    ses = re.search(r"ses-(\d+)", filename_or_path)
    run = re.search(r"run-(\d+)", filename_or_path)
    if sub is None or ses is None:
        return None
    key = f"sub-{sub.group(1)}_ses-{ses.group(1)}"
    if run is not None:
        key += f"_run-{run.group(1)}"
    return key


def _pair_volumes_and_masks(volumes, masks):
    """Pair volume files to mask files by their join key.

    The join key is built by ``_extract_join_key`` from the volume *filename*
    and the mask *parent folder name*.  For dataset_RU the key is
    ``sub-XXX_ses-Y_run-Z`` (all three components present in both names).
    For dataset_T1 and older datasets where the volume filename has no run
    component, the key falls back to ``sub-XXX_ses-Y``.

    Any volume or mask that cannot be matched is reported and silently
    dropped — no silent mispairs.
    """
    vol_by_key = {}
    for v in volumes:
        key = _extract_join_key(op.basename(v))
        if key is None:
            continue
        if key in vol_by_key:
            print(f"[create_dataset] WARNING: duplicate volume key '{key}', "
                  f"keeping {op.basename(vol_by_key[key])}, "
                  f"skipping {op.basename(v)}")
        else:
            vol_by_key[key] = v

    mask_by_key = {}
    for m in masks:
        # The key lives in the mask's parent folder name, e.g. sub-001_ses-3_run-01
        key = _extract_join_key(op.basename(op.dirname(m)))
        if key is None:
            continue
        if key in mask_by_key:
            print(f"[create_dataset] WARNING: duplicate mask key '{key}', "
                  f"keeping first, skipping {m}")
        else:
            mask_by_key[key] = m

    common_keys = sorted(set(vol_by_key).intersection(mask_by_key))
    paired_vols = [vol_by_key[k] for k in common_keys]
    paired_masks = [mask_by_key[k] for k in common_keys]

    orphan_vols = sorted(set(vol_by_key) - set(mask_by_key))
    orphan_masks = sorted(set(mask_by_key) - set(vol_by_key))
    if orphan_vols:
        print(f"[create_dataset] dropped {len(orphan_vols)} volume(s) with no "
              f"matching mask: {orphan_vols[:5]}{'...' if len(orphan_vols) > 5 else ''}")
    if orphan_masks:
        print(f"[create_dataset] dropped {len(orphan_masks)} mask(s) with no "
              f"matching volume: {orphan_masks[:5]}{'...' if len(orphan_masks) > 5 else ''}")

    return paired_vols, paired_masks


def create_dataset(
    subset,
    validation_cases,
    test_cases,
    seed=42,
    verbose=False,
    dataset_root=None,
    split_mode="subject",
    folds=0,
    fold=0,
):
    """Build a MONAI ArrayDataset for one split.

    Works with all known dataset layouts under the project:

      dataset_RU  (current)  rsub-XXX_ses-Y_acq-petra_run-Z_PDw.nii
      dataset_T1             rsub-XXX_ses-Y_acq-mprage_T1w.nii
      dataset / dataset_cut  rsub-XXX_ses-Y_acq-*.nii (original paper)

    The glob ``rsub*.nii`` intentionally catches all of the above.  Pairing
    is done on the most specific available key (sub+ses+run when the volume
    filename carries a run component, sub+ses otherwise) so every layout
    is handled correctly without separate code paths.

    Parameters
    ----------
    subset : {"train", "validation", "test"}
        Which split to return.
    validation_cases, test_cases : int
        Number of cases held out for validation / test.
    seed : int
        Seed for the deterministic split (independent of train-time RNG).
    verbose : bool
        If True, print the shape of every loaded (volume, mask) pair.
    dataset_root : str | None
        Optional explicit dataset root. If omitted, the root is resolved
        via ``resolve_dataset_root()`` — which checks the
        ``ELECTRODE_DATASET_ROOT`` env var then the built-in candidate list
        (dataset_RU first, then the older fallbacks).
    """
    mask_suffix = "mask.nii.gz"

    root_dir = resolve_dataset_root(dataset_root)
    print(f"[create_dataset] using dataset root: {root_dir}")

    subject_pattern_vol  = op.join(root_dir, "sub-*", "unzipped")
    subject_pattern_mask = op.join(root_dir, "sub-*", "unzipped", "sub-*")

    # rsub*.nii matches every known volume naming convention across all
    # dataset variants (T1w, PDw, petra, mprage) without needing separate globs.
    volumes = sorted(glob(op.join(subject_pattern_vol, "rsub*.nii")))
    masks   = sorted(glob(op.join(subject_pattern_mask, mask_suffix)))

    if not volumes:
        raise FileNotFoundError(
            f"No MRI volumes matching 'rsub*.nii' found under {root_dir}. "
            "Check the dataset path or set ELECTRODE_DATASET_ROOT."
        )
    if not masks:
        raise FileNotFoundError(
            f"No masks matching {mask_suffix} found under {root_dir}."
        )

    # Pair by sub/ses key (NOT by sort order -- the original paper code
    # had a latent bug whenever a subject had multiple sessions).
    volumes, masks = _pair_volumes_and_masks(volumes, masks)
    print(f"[create_dataset] {len(volumes)} paired (volume, mask) cases available.")

    total_needed = validation_cases + test_cases
    if len(volumes) <= total_needed:
        raise ValueError(
            f"Not enough cases: have {len(volumes)} paired, but validation"
            f"+test alone need {total_needed}."
        )

    volumes, masks = subsetting(
        subset=subset, vols=volumes, mask=masks,
        validation_cases=validation_cases, test_cases=test_cases, seed=seed,
        split_mode=split_mode, folds=folds, fold=fold,
        verbose=(subset == "train"),   # print the split banner once, not 3x
    )
    SPLIT_MANIFEST.setdefault("subjects", {})[subset] = sorted(
        {extract_subject_id(v) for v in volumes})
    SPLIT_MANIFEST.setdefault("files", {})[subset] = [str(v) for v in volumes]
    print(f"[create_dataset] subset='{subset}': {len(volumes)} cases.")

    if verbose:
        for vol, mask in zip(volumes, masks):
            vol_data = tfms.LoadImage(image_only=True)(vol)
            mask_data = tfms.LoadImage(image_only=True)(mask)
            print(f"Volume shape: {vol_data.shape}, Mask shape: {mask_data.shape}")

    vol_tfms = tfms.Compose([
        tfms.LoadImage(image_only=True),
        tfms.ScaleIntensity(),
        tfms.EnsureChannelFirst(),
        tfms.RandZoom(1, min_zoom=0.7, max_zoom=1.3),
        tfms.RandRotate(prob=1, range_x=0.5, range_y=0.5, range_z=0.5, keep_size=True),
        tfms.RandAffine(prob=1, rotate_range=0.5, shear_range=0.5, padding_mode="zeros"),
        tfms.Resize((224, 288, 288)),
        tfms.RandFlip(prob=0.5, spatial_axis=0),
        tfms.RandGaussianNoise(prob=0.1, mean=0.0, std=0.1),
        tfms.RandGaussianSmooth(prob=0.1, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5)),
        tfms.RandAdjustContrast(prob=0.1, gamma=(0.5, 2.0)),
        tfms.RandShiftIntensity(offsets=0.1, prob=0.01),
        tfms.RandCoarseDropout(holes=10, spatial_size=5, fill_value=None, prob=0.01),
    ])

    mask_tfms = tfms.Compose([
        tfms.LoadImage(image_only=True),
        tfms.EnsureChannelFirst(),
        tfms.RandZoom(1, min_zoom=0.7, max_zoom=1.3, mode="nearest"),
        tfms.RandRotate(prob=1, range_x=0.5, range_y=0.5, range_z=0.5, keep_size=True, mode="nearest"),
        tfms.RandAffine(prob=1, rotate_range=0.5, shear_range=0.5, padding_mode="zeros", mode="nearest"),
        tfms.Resize((224, 288, 288)),
        tfms.RandFlip(prob=0.5, spatial_axis=0),
        # NOTE: RandShiftIntensity and RandCoarseDropout were removed here.
        # They are intensity augmentations and must never touch a label map:
        # ShiftIntensity turns the 0/1 labels into 0.1/1.1, and CoarseDropout
        # with fill_value=None punches holes of random values into the ground
        # truth. Both corrupt the target the loss is computed against. They
        # remain in vol_tfms, where they belong.
        #
        # Geometry stays synchronised with the volume because MONAI's
        # ArrayDataset draws ONE seed per item and calls set_random_state() on
        # both chains: the first five transforms (RandZoom, RandRotate,
        # RandAffine, Resize, RandFlip) are identical and in the same positions
        # in both lists, so they consume the same randomness. The image-only
        # intensity transforms come after them and therefore cannot desync the
        # geometry.
    ])

    return ArrayDataset(volumes, vol_tfms, masks, mask_tfms)


# --------------------------------------------------------------------------- #
# Network wrapper: training / validation / test loops
# --------------------------------------------------------------------------- #
class Network:
    def __init__(
        self,
        net,
        scaler,
        opt,
        loss_function,
        train_loader,
        val_loader,
        test_loader,
        dice_metric,
        hausdorff_metric,
        iou_metric,
        eval_num,
        max_iterations,
        root_dir,
        accumulation_steps=4,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.2,
        writer=None,
        ckpt_name="best_metric_model.pth",
        debug_tag="exp",
        # Sanity-check / early-stop knobs (see _sanity_check() below).
        # These exist to abort a training run that has clearly broken so we
        # don't burn 42k iterations after the loss has already diverged.
        min_dice_threshold=0.05,
        sanity_probation_iters=15_000,   # raised: see _sanity_check_eval docstring
        max_no_improve_evals=40,         # raised: see _sanity_check_eval docstring
    ):
        self.net = net
        self.scaler = scaler
        # Set by run_experiment(); defaults keep standalone use working.
        self.amp_dtype = getattr(self, "amp_dtype", torch.bfloat16)
        self.grad_clip = getattr(self, "grad_clip", 1.0)
        self.opt = opt
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.dice_metric = dice_metric
        self.hausdorff_metric = hausdorff_metric
        self.iou_metric = iou_metric
        self.eval_num = eval_num
        self.max_iterations = max_iterations
        self.epoch_loss_values = []
        self.metric_values = []
        self.dice_val_best = 0.0
        self.hausdorff_at_best = float("nan")
        self.iou_at_best = float("nan")
        self.global_step_best = 0
        self.global_step = 0
        self.root_dir = root_dir
        self.accumulation_steps = accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.writer = writer
        self.ckpt_name = ckpt_name
        self.debug_tag = debug_tag
        # --- LR schedule: linear warmup -> cosine-annealing-with-restarts ---
        # NB: scheduler.step() is called once per OPTIMIZER step (every
        # accumulation_steps micro-batches), NOT every micro-batch -- see the
        # train loop. So warmup_steps below is counted in OPTIMIZER steps.
        #
        # Warmup ramps the LR linearly from (start_factor * base_lr) up to the
        # full base_lr over warmup_steps, then hands off to the original cosine
        # schedule. This lets you use a higher target LR without the early-
        # iteration divergence (the NaN-at-iter-37 problem) because the first
        # few hundred updates are gentle.
        warmup_steps = 200          # optimizer steps of warmup; tune as needed
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = CosineAnnealingWarmRestarts(self.opt, T_0=1000, T_mult=1, eta_min=1e-5)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.opt, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

        # Sanity-check state
        self.min_dice_threshold = min_dice_threshold
        self.sanity_probation_iters = sanity_probation_iters
        self.max_no_improve_evals = max_no_improve_evals
        self._evals_since_improvement = 0
        self._n_nan_loss = 0
        self.failure_reason = None     # set if a sanity check trips
        self.stop_training = False     # train loop checks this on every step

    def _sanity_check_loss(self, loss_value):
        """Detect divergence: abort only after 5 consecutive NaN/Inf losses.

        Rationale from observed training behaviour:
          - AMP + DiceFocal loss can produce isolated NaN/inf values during
            the first ~2,000 iterations while the network output is near-random
            and the focal term amplifies instability.  The GradScaler skips the
            weight update on those steps and the run recovers cleanly.
          - A truly dead run produces *sustained* non-finite loss, not isolated
            spikes.  Requiring 5 in a row (instead of 2) avoids aborting a run
            that would have recovered.
          - The counter decrements by 1 on each good step (instead of resetting
            to 0) so that an alternating NaN/finite pattern is still caught
            rather than being reset on every single good step.
        """
        if not np.isfinite(loss_value):
            self._n_nan_loss += 1
            if self._n_nan_loss >= 5:
                self.failure_reason = (
                    f"Loss diverged: 5 consecutive non-finite loss values "
                    f"(latest={loss_value}) at iter {self.global_step}. "
                    f"Try lowering --lr or removing num_res_units."
                )
                self.stop_training = True
        else:
            # Decrement instead of zero-reset so alternating NaN/finite
            # patterns (NaN, ok, NaN, ok, ...) still accumulate toward abort.
            self._n_nan_loss = max(0, self._n_nan_loss - 1)

    def _sanity_check_eval(self, dice_val, improved):
        """Detect a model that is clearly not learning — without aborting
        a slow-starting run like the paper's AttentionUnet.

        Abort only when ALL THREE conditions hold simultaneously:
          1. Past the probation window (default 15,000 iters) — long enough
             for the DiceFocal loss to escape the near-random-output phase
             that produces inf Hausdorff in the first ~2,000 steps.
          2. The best-ever validation Dice has never exceeded
             `min_dice_threshold` (default 0.05) — i.e. the model has truly
             never learned anything at all.
          3. No improvement for `max_no_improve_evals` consecutive evaluations
             (default 40, = 20,000 iters at eval_num=500) — a generous window
             that allows the noisy plateau-and-jump pattern observed in this
             network (e.g. Dice 0.25 at step 20,000 then 0.64 at step 20,500).

        The combination of conditions 2 + 3 means:
          - A run that crossed 0.05 even once is NEVER aborted by this check,
            no matter how long it stagnates afterwards.  Use --max-iterations
            to set a hard wall instead.
          - A run that never reached 0.05 is only aborted after both the
            probation window and the no-improvement window have expired.
        """
        if improved:
            self._evals_since_improvement = 0
        else:
            self._evals_since_improvement += 1

        past_probation = self.global_step >= self.sanity_probation_iters
        never_learned = self.dice_val_best < self.min_dice_threshold
        stagnant = self._evals_since_improvement >= self.max_no_improve_evals

        if past_probation and never_learned and stagnant:
            self.failure_reason = (
                f"Model is not learning: best validation Dice = "
                f"{self.dice_val_best:.4f} after {self.global_step} iters, "
                f"no improvement for {self._evals_since_improvement} "
                f"evaluations (={self._evals_since_improvement * 500} iters). "
                f"Likely a training-stability problem "
                f"(divergence, dead activations, wrong LR for this "
                f"architecture)."
            )
            self.stop_training = True

    def train(self):
        self.net.train()
        epoch_loss = 0.0
        step = 0
        epoch_iterator = tqdm(self.train_loader, desc="Training", dynamic_ncols=True)

        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = batch[0].cuda(), batch[1].cuda()

            # Forward in reduced precision, but compute the LOSS IN FP32.
            # DiceFocalLoss uses smooth_nr/smooth_dr = 1e-6, which is below
            # float16's smallest normal value (~6e-5). Under fp16 autocast those
            # terms underflow to zero, so a batch whose foreground vanishes
            # after augmentation gives 0/0 -> NaN. Folds 0 and 1 of the
            # c_increased CV run died exactly this way (5 consecutive NaN at
            # iter ~4000) while folds 2-4 survived -- the classic signature of a
            # precision problem that only some data orderings trigger.
            with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                logit_map = self.net(x)
            loss = self.loss_function(logit_map.float(), y) / self.accumulation_steps

            # Sanity check 1: detect loss divergence
            loss_value = loss.item() * self.accumulation_steps
            self._sanity_check_loss(loss_value)
            if self.stop_training:
                print(f"\n[sanity-check] ABORT: {self.failure_reason}")
                break

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accumulation_steps == 0:
                # unscale_ before clipping, otherwise the norm is computed on
                # scaler-inflated gradients and the clip threshold is meaningless.
                if self.grad_clip and self.grad_clip > 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                                   self.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.scheduler.step()
                self.opt.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * self.accumulation_steps

            if self.global_step % 300 == 0:
                with torch.no_grad():
                    pred = torch.argmax(logit_map.detach(), dim=1).unsqueeze(1)
                    save_debug_batch(x.detach(), y.detach(), pred, tag=self.debug_tag)

            epoch_iterator.set_description(
                f"Training ({self.global_step}/{self.max_iterations}) "
                f"loss={loss.item() * self.accumulation_steps:.5f}"
            )

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", loss.item() * self.accumulation_steps, self.global_step)
                self.writer.add_scalar("LR", self.opt.param_groups[0]["lr"], self.global_step)

            if (self.global_step % self.eval_num == 0) or (self.global_step == self.max_iterations):
                dice_val, hausdorff_val, iou_val = self.validation()
                epoch_loss /= max(step, 1)
                self.epoch_loss_values.append(epoch_loss)
                self.metric_values.append(dice_val)

                if self.writer is not None:
                    self.writer.add_scalar("Dice/val", dice_val, self.global_step)
                    self.writer.add_scalar("Hausdorff/val", hausdorff_val, self.global_step)
                    self.writer.add_scalar("IoU/val", iou_val, self.global_step)

                if dice_val > self.dice_val_best:
                    self.dice_val_best = dice_val
                    self.hausdorff_at_best = hausdorff_val
                    self.iou_at_best = iou_val
                    self.global_step_best = self.global_step
                    torch.save(self.net.state_dict(), op.join(self.root_dir, self.ckpt_name))
                    print(f"Model saved — best Dice: {self.dice_val_best:.4f}")
                    improved = True
                else:
                    improved = False

                # Sanity check 2: detect a model that never crosses the
                # min-Dice threshold long after it should have done so.
                self._sanity_check_eval(dice_val, improved)
                if self.stop_training:
                    print(f"\n[sanity-check] ABORT: {self.failure_reason}")
                    break

            self.global_step += 1

    @torch.no_grad()
    def validation(self):
        post_pred = tfms.Compose([tfms.AsDiscrete(argmax=True, to_onehot=2)])
        post_label = tfms.Compose([tfms.AsDiscrete(to_onehot=2)])
        self.net.eval()
        dice_values, hausdorff_values, iou_values = [], [], []

        for batch in self.val_loader:
            val_inputs, val_labels = batch[0].cuda(), batch[1].cuda()
            val_output = self.net(val_inputs)

            val_output_ = [post_pred(i) for i in decollate_batch(val_output)]
            val_labels_ = [post_label(i) for i in decollate_batch(val_labels)]

            self.dice_metric(y_pred=val_output_, y=val_labels_)
            self.hausdorff_metric(y_pred=val_output_, y=val_labels_)
            self.iou_metric(y_pred=val_output_, y=val_labels_)

            dice_values.append(self.dice_metric.aggregate().item())
            hausdorff_values.append(self.hausdorff_metric.aggregate().item())
            iou_values.append(self.iou_metric.aggregate().item())

            self.dice_metric.reset()
            self.hausdorff_metric.reset()
            self.iou_metric.reset()

        d, h, i = float(np.mean(dice_values)), float(np.mean(hausdorff_values)), float(np.mean(iou_values))
        print(f"Validation - Dice: {d:.4f}, Hausdorff: {h:.4f}, IoU: {i:.4f}")
        return d, h, i

    @torch.no_grad()
    def test(self):
        post_pred = tfms.Compose([tfms.AsDiscrete(argmax=True, to_onehot=2)])
        post_label = tfms.Compose([tfms.AsDiscrete(to_onehot=2)])
        self.net.eval()
        dice_values, hausdorff_values, iou_values = [], [], []

        for batch in self.test_loader:
            test_inputs, test_labels = batch[0].cuda(), batch[1].cuda()
            test_output = self.net(test_inputs)

            test_output_ = [post_pred(i) for i in decollate_batch(test_output)]
            test_labels_ = [post_label(i) for i in decollate_batch(test_labels)]

            self.dice_metric(y_pred=test_output_, y=test_labels_)
            self.hausdorff_metric(y_pred=test_output_, y=test_labels_)
            self.iou_metric(y_pred=test_output_, y=test_labels_)

            dice_values.append(self.dice_metric.aggregate().item())
            hausdorff_values.append(self.hausdorff_metric.aggregate().item())
            iou_values.append(self.iou_metric.aggregate().item())

            self.dice_metric.reset()
            self.hausdorff_metric.reset()
            self.iou_metric.reset()

        # ---- per-case metrics ------------------------------------------
        # batch_size=1 and the test loader is unshuffled, so the i-th value is
        # the i-th test file. One row per image makes results auditable and
        # lets metrics be recomputed per participant without retraining.
        _files = SPLIT_MANIFEST.get("files", {}).get("test", [])
        if _files and len(_files) == len(dice_values):
            _csv = op.join(self.root_dir, f"{self.debug_tag}_test_per_case.csv")
            with open(_csv, "w", newline="") as _fh:
                _w = csv.writer(_fh)
                _w.writerow(["file", "subject", "dice", "hausdorff", "iou"])
                for _f, _d, _h, _i in zip(_files, dice_values,
                                          hausdorff_values, iou_values):
                    _w.writerow([op.basename(_f), extract_subject_id(_f), _d, _h, _i])
            print(f"[test] per-case metrics -> {_csv}")

        d = float(np.mean(dice_values))
        h = float(np.mean(hausdorff_values))
        i = float(np.mean(iou_values))
        d_std = float(np.std(dice_values))
        h_std = float(np.std(hausdorff_values))
        i_std = float(np.std(iou_values))

        if self.writer is not None:
            self.writer.add_scalar("Dice/test", d, self.global_step)
            self.writer.add_scalar("Hausdorff/test", h, self.global_step)
            self.writer.add_scalar("IoU/test", i, self.global_step)

        print(f"Test Results — Dice: {d:.4f}±{d_std:.4f}, "
              f"Hausdorff: {h:.4f}±{h_std:.4f}, IoU: {i:.4f}±{i_std:.4f}")
        return {
            "dice_mean": d, "dice_std": d_std,
            "hausdorff_mean": h, "hausdorff_std": h_std,
            "iou_mean": i, "iou_std": i_std,
            "dice_values": dice_values,
            "hausdorff_values": hausdorff_values,
            "iou_values": iou_values,
        }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def count_parameters(net):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total, trainable


def write_report(
    experiment_tag,
    experiment_description,
    model_config,
    train_config,
    param_counts,
    best_val,
    test_metrics,
    results_dir=RESULTS_DIR_DEFAULT,
):
    """Write a Markdown and a TXT report for one ablation experiment."""
    os.makedirs(results_dir, exist_ok=True)
    md_path = op.join(results_dir, f"{experiment_tag}_report.md")
    txt_path = op.join(results_dir, f"{experiment_tag}_metrics.txt")
    json_path = op.join(results_dir, f"{experiment_tag}_metrics.json")

    total_params, trainable_params = param_counts
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append(f"# Ablation experiment — {experiment_tag}\n")
    md.append(f"*Generated:* {now}  \n")
    md.append(f"*Host:* {platform.node()}  \n")
    md.append(f"*PyTorch:* {torch.__version__}  \n")
    if torch.cuda.is_available():
        md.append(f"*GPU:* {torch.cuda.get_device_name(0)}  \n")

    # Failure banner: make it impossible to miss that this run is invalid.
    if test_metrics.get("FAILED"):
        md.append("\n> **⚠ TRAINING FAILED — DO NOT USE THESE NUMBERS IN THE MANUSCRIPT.**\n>\n")
        md.append(f"> {test_metrics.get('failure_reason', 'unknown reason')}\n")

    md.append("\n## Description\n")
    md.append(f"{experiment_description}\n")

    md.append("\n## Model configuration\n")
    md.append("| Parameter | Value |\n|---|---|\n")
    for k, v in model_config.items():
        md.append(f"| `{k}` | {v} |\n")
    md.append(f"| **Total parameters** | {total_params:,} |\n")
    md.append(f"| **Trainable parameters** | {trainable_params:,} |\n")

    md.append("\n## Training configuration\n")
    md.append("| Parameter | Value |\n|---|---|\n")
    for k, v in train_config.items():
        md.append(f"| `{k}` | {v} |\n")

    md.append("\n## Best validation checkpoint\n")
    md.append("| Metric | Value |\n|---|---|\n")
    md.append(f"| Best Dice (val) | {best_val['dice']:.4f} |\n")
    md.append(f"| Hausdorff @ best Dice (val) | {best_val['hausdorff']:.4f} |\n")
    md.append(f"| IoU @ best Dice (val) | {best_val['iou']:.4f} |\n")
    md.append(f"| Iteration of best checkpoint | {best_val['iteration']} |\n")

    md.append("\n## Held-out test set results\n")
    md.append("| Metric | Mean | Std |\n|---|---|---|\n")
    md.append(f"| Dice | {test_metrics['dice_mean']:.4f} | {test_metrics['dice_std']:.4f} |\n")
    md.append(f"| Hausdorff | {test_metrics['hausdorff_mean']:.4f} | {test_metrics['hausdorff_std']:.4f} |\n")
    md.append(f"| IoU | {test_metrics['iou_mean']:.4f} | {test_metrics['iou_std']:.4f} |\n")

    md.append("\n### Per-sample test metrics\n")
    md.append("| Sample | Dice | Hausdorff | IoU |\n|---|---|---|---|\n")
    for idx, (d, h, i) in enumerate(zip(
        test_metrics["dice_values"],
        test_metrics["hausdorff_values"],
        test_metrics["iou_values"],
    )):
        md.append(f"| {idx} | {d:.4f} | {h:.4f} | {i:.4f} |\n")

    with open(md_path, "w") as f:
        f.writelines(md)

    # Plain-text summary (for easy grep / paste into the manuscript)
    with open(txt_path, "w") as f:
        f.write(f"Experiment: {experiment_tag}\n")
        f.write(f"Description: {experiment_description}\n")
        f.write(f"Generated: {now}\n\n")
        f.write("Model configuration:\n")
        for k, v in model_config.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"  total_parameters: {total_params}\n")
        f.write(f"  trainable_parameters: {trainable_params}\n\n")
        f.write("Training configuration:\n")
        for k, v in train_config.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nBest validation:\n")
        f.write(f"  dice: {best_val['dice']:.4f}\n")
        f.write(f"  hausdorff: {best_val['hausdorff']:.4f}\n")
        f.write(f"  iou: {best_val['iou']:.4f}\n")
        f.write(f"  iteration: {best_val['iteration']}\n\n")
        f.write("Test set (held-out):\n")
        f.write(f"  dice_mean: {test_metrics['dice_mean']:.4f}\n")
        f.write(f"  dice_std:  {test_metrics['dice_std']:.4f}\n")
        f.write(f"  hausdorff_mean: {test_metrics['hausdorff_mean']:.4f}\n")
        f.write(f"  hausdorff_std:  {test_metrics['hausdorff_std']:.4f}\n")
        f.write(f"  iou_mean: {test_metrics['iou_mean']:.4f}\n")
        f.write(f"  iou_std:  {test_metrics['iou_std']:.4f}\n")

    # JSON for downstream aggregation
    payload = {
        "experiment_tag": experiment_tag,
        "description": experiment_description,
        "generated": now,
        "model_config": {k: str(v) for k, v in model_config.items()},
        "train_config": {k: str(v) for k, v in train_config.items()},
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "best_val": best_val,
        "test_metrics": test_metrics,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Report written:\n  {md_path}\n  {txt_path}\n  {json_path}")
    return md_path, txt_path, json_path


# --------------------------------------------------------------------------- #
# CLI helper -- shared by a_, b_, c_ scripts
# --------------------------------------------------------------------------- #
def build_common_argparser(description=None):
    """Build the argparse.ArgumentParser shared by all ablation scripts.

    Sub-scripts can grab extra arguments by calling
        parser = build_common_argparser(description=...)
        parser.add_argument("--my-extra", ...)
        args = parser.parse_args()

    The flags exposed here are the things you most often need to change at
    the command line when re-running on a new machine or for a quick smoke
    test (dataset path, iteration count, etc.).
    """
    import argparse
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset-root", default=None,
        help="Override the dataset root. Default: $ELECTRODE_DATASET_ROOT "
             "or the first existing path among the built-in candidates "
             "(currently the dataset_T1 path on /media/data04, with the "
             "original /media/data01 path as fallback).",
    )
    p.add_argument("--max-iterations", type=int, default=50_000)
    p.add_argument("--eval-num", type=int, default=500)
    p.add_argument("--seed", type=int, default=1001)
    p.add_argument(
        "--amp-dtype", choices=["bf16", "fp16", "off"], default="bf16",
        help="Autocast precision for the forward pass. 'bf16' (default) has "
             "the same exponent range as fp32 and is the right choice on A100 "
             "-- it removes the overflow/underflow that produced NaN losses "
             "with fp16. 'fp16' reproduces the old behaviour. 'off' runs in "
             "full fp32 (slowest, most stable). The loss is always computed "
             "in fp32 regardless of this setting.",
    )
    p.add_argument(
        "--grad-clip", type=float, default=1.0,
        help="Max gradient norm (0 disables). Guards against the loss spikes "
             "that precede divergence.",
    )
    p.add_argument(
        "--dataset", choices=["HGW", "RU"], default=None,
        help="Shorthand for --dataset-root $HOME/Dataset/dataset_<NAME>. "
             "HGW = 54 images / 38 participants (the manuscript ground-truth "
             "set). RU = 72 images / 52 participants. Both use identical "
             "file naming; RU additionally contains 4-digit subject IDs. "
             "The chosen name is added to the output tag so HGW and RU runs "
             "never overwrite each other. --dataset-root overrides this.",
    )
    p.add_argument(
        "--split-mode", choices=["subject", "legacy_image", "final"], default="subject",
        help="'subject' (default): whole participants go to exactly one of "
             "train/val/test -- no leakage. 'legacy_image': the original "
             "image-wise shuffle, for reproducing published results only. "
             "'final': train the deployable model on every participant not "
             "held out for validation -- NO test set. Use this after the CV "
             "ablation has already told you which architecture to ship and "
             "what performance to expect; report the CV mean, not a number "
             "from this run. Incompatible with --folds.",
    )
    p.add_argument(
        "--folds", type=int, default=0,
        help="Number of subject-wise cross-validation folds. 0 (default) = "
             "single train/val/test split. Use 5 for 5-fold CV; then "
             "--validation-cases/--test-cases are ignored.",
    )
    p.add_argument(
        "--fold", type=int, default=0,
        help="Which fold to run (0-indexed) when --folds >= 2. In SLURM use "
             "--fold $SLURM_ARRAY_TASK_ID with --array=0-4.",
    )
    p.add_argument("--validation-cases", type=int, default=8)
    p.add_argument("--test-cases", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulation-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.005)
    # DataLoader robustness knobs (see _configure_multiprocessing_for_large_volumes)
    p.add_argument(
        "--num-workers", type=int, default=2,
        help="DataLoader worker processes. Default 2 (was 4 in the original "
             "paper code; 4 caused 'received 0 items of ancdata' on hosts "
             "with default Linux FD limits). Set to 0 for single-process "
             "loading if you keep hitting FD issues.",
    )
    p.add_argument(
        "--no-pin-memory", action="store_true",
        help="Disable pin_memory in the DataLoader. Useful when the "
             "pin-memory thread keeps dying from FD exhaustion.",
    )
    p.add_argument("--prefetch-factor", type=int, default=2)
    # Sanity-check / early-stop knobs (Network._sanity_check_*).
    p.add_argument(
        "--min-dice-threshold", type=float, default=0.05,
        help="Validation Dice the model must exceed at least once to be "
             "considered 'learning'. A run that never crosses this is "
             "aborted after --sanity-probation-iters + "
             "--max-no-improve-evals*eval_num iters. Default 0.05.",
    )
    p.add_argument(
        "--sanity-probation-iters", type=int, default=15_000,
        help="Training iterations during which no eval-based sanity check "
             "is enforced.  Raised from 10,000 to 15,000 because the "
             "AttentionUnet with DiceFocal loss can take ~3,000 iters to "
             "escape near-random outputs (inf Hausdorff, Dice<0.05) before "
             "it starts learning.  Default 15,000.",
    )
    p.add_argument(
        "--max-no-improve-evals", type=int, default=40,
        help="If best validation Dice is STILL below --min-dice-threshold "
             "AND no improvement for this many evaluations, abort.  "
             "Raised from 20 to 40 (=20,000 iters at eval_num=500) because "
             "this network shows a noisy plateau-and-jump pattern: Dice can "
             "sit near 0.25 for thousands of steps then jump to 0.64.  "
             "Once the model ever exceeds --min-dice-threshold this check "
             "is permanently disabled for that run.  Set to a very large "
             "value to fully disable.",
    )
    p.add_argument(
        "--optimizer",
        choices=["paper", "adamax", "adam_paperbc3d"],
        default="paper",
        help="Which optimiser recipe to use. 'paper' (default) is "
             "torch.optim.Adam(net.parameters()) with all defaults — the "
             "recipe used by model.py to produce the published numbers, "
             "and the recipe that trains the full-sized AttentionUnet "
             "stably. 'adamax' is the old Adamax(lr=5e-3, betas=(0.95, "
             "0.99), wd=1e-5) recipe — kept only for backwards-compat "
             "reproduction of earlier (a) and (c_reduced) runs. "
             "'adam_paperbc3d' is the recipe used by model_bc_3d.py "
             "(Adam, lr=1e-2, weight_decay=1e-4). Pair with "
             "--loss diceloss to reproduce that paper run exactly.",
    )
    p.add_argument(
        "--loss",
        choices=["dicefocal", "dicece", "diceloss"],
        default="dicefocal",
        help="Which loss to use. 'dicefocal' (default) is DiceFocalLoss"
             "(lambda_dice=0.3, lambda_focal=0.7, gamma=2.5, "
             "include_background=False) — the loss under which the (a), (b), "
             "(c), (d) ablations were trained. 'diceloss' is plain "
             "DiceLoss(to_onehot_y=True, softmax=True) — the loss used by "
             "model_bc_3d.py. 'dicece' is DiceCELoss(lambda_dice=0.5, "
             "lambda_ce=0.5, ce_weight=[1,10]) -- weights the electrode class "
             "10x in the cross-entropy term to counter the all-background "
             "collapse seen under 'dicefocal'.",
    )
    p.add_argument(
        "--tag-suffix", type=str, default="",
        help="Optional string appended to the experiment_tag, so that "
             "different runs/variants/seeds write to DIFFERENT checkpoint, "
             "report, and TensorBoard files instead of overwriting each "
             "other. E.g. two AttentionUnet variants that share the "
             "hardcoded tag 'paper_attention_unet' will collide unless you "
             "pass distinct --tag-suffix values (e.g. '_normal' vs "
             "'_increased'). Also useful for multiple seeds: '_seed1001'. "
             "A leading underscore is added automatically if you omit it.",
    )
    return p


def run_experiment_from_args(
    args, *, experiment_tag, experiment_description, build_model, model_config_summary
):
    """Convenience wrapper that maps argparse Namespace -> run_experiment()."""
    # Disambiguate outputs across runs/variants/seeds. Without this, two
    # scripts that hardcode the same EXPERIMENT_TAG (e.g. c_normal and
    # c_increased both use "paper_attention_unet") overwrite each other's
    # checkpoint, report, and TensorBoard dir. --tag-suffix is opt-in and
    # defaults to "" (no change), so existing behaviour is preserved.
    suffix = getattr(args, "tag_suffix", "") or ""
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix

    if getattr(args, "split_mode", None) == "final" and getattr(args, "folds", 0):
        raise ValueError("--split-mode final trains one deployable model and is "
                         "incompatible with --folds (that's what the CV run was for).")

    # --dataset HGW|RU -> concrete root. resolve_dataset_root() already walks
    # down to the sub-* directory, so the top-level folder is enough.
    if getattr(args, "dataset", None) and not args.dataset_root:
        args.dataset_root = op.join(
            op.expanduser("~"), "Dataset", f"dataset_{args.dataset}")

    # AUTOMATIC output disambiguation. Every job writes a tag that encodes the
    # dataset, the split, the fold and the seed, so parallel SLURM jobs can
    # never overwrite each other's checkpoint / report / TensorBoard dir --
    # even when two scripts hardcode the same EXPERIMENT_TAG (c_normal and
    # c_increased both use "paper_attention_unet").
    auto = ""
    if getattr(args, "dataset", None):
        auto += f"_{args.dataset}"
    if getattr(args, "folds", 0) and args.folds >= 2:
        auto += f"_cv{args.folds}f{args.fold}"
    elif getattr(args, "split_mode", "subject") == "final":
        # Must be marked: a final-model run has NO test set, so its outputs
        # must never be mistaken for a single-split run that does.
        auto += "_FINAL"
    elif getattr(args, "split_mode", "subject") == "legacy_image":
        auto += "_legacysplit"
    auto += f"_seed{args.seed}"
    experiment_tag = f"{experiment_tag}{auto}{suffix}"
    print(f"[tag] outputs will be written under: {experiment_tag}")

    return run_experiment(
        experiment_tag=experiment_tag,
        experiment_description=experiment_description,
        build_model=build_model,
        model_config_summary=model_config_summary,
        vc=args.validation_cases,
        tc=args.test_cases,
        bs=args.batch_size,
        seed=args.seed,
        accumulation_steps=args.accumulation_steps,
        max_iterations=args.max_iterations,
        eval_num=args.eval_num,
        lr=args.lr,
        dataset_root=args.dataset_root,
        num_workers=args.num_workers,
        pin_memory=False if args.no_pin_memory else None,
        prefetch_factor=args.prefetch_factor,
        min_dice_threshold=args.min_dice_threshold,
        sanity_probation_iters=args.sanity_probation_iters,
        max_no_improve_evals=args.max_no_improve_evals,
        optimizer_name=args.optimizer,
        loss_name=args.loss,
        split_mode=args.split_mode,
        folds=args.folds,
        fold=args.fold,
        amp_dtype=getattr(args, "amp_dtype", "bf16"),
        grad_clip=getattr(args, "grad_clip", 1.0),
    )


# --------------------------------------------------------------------------- #
# Top-level experiment runner — used by a_, b_, c_ scripts
# --------------------------------------------------------------------------- #
def run_experiment(
    *,
    experiment_tag,
    experiment_description,
    build_model,
    model_config_summary,
    vc=8, tc=8, bs=1, seed=1001,
    accumulation_steps=8,
    max_iterations=50_000,
    eval_num=500,
    lr=0.005,
    results_dir=RESULTS_DIR_DEFAULT,
    runs_dir=RUNS_DIR_DEFAULT,
    dataset_root=None,
    num_workers=2,
    pin_memory=None,
    prefetch_factor=2,
    min_dice_threshold=0.05,
    sanity_probation_iters=15_000,   # raised: AttentionUnet needs ~3k iters to escape NaN/inf phase
    max_no_improve_evals=40,         # raised: noisy plateau-and-jump pattern needs more patience
    optimizer_name="paper",
    loss_name="dicefocal",
    split_mode="subject",
    folds=0,
    fold=0,
    amp_dtype="bf16",
    grad_clip=1.0,
):
    """End-to-end run for one ablation variant.

    `build_model` is a zero-argument callable that returns the constructed
    `torch.nn.Module`. `model_config_summary` is a dict of the hyper-parameters
    we want recorded in the Markdown / TXT report.

    `dataset_root` -- optional override for the dataset location. If omitted,
    `resolve_dataset_root()` is used (env var or built-in candidates).

    DataLoader settings -- `num_workers`, `pin_memory`, `prefetch_factor`
    default to a conservative configuration that does NOT exhaust Linux FD
    limits on hosts with 3D volumes (224x288x288 etc.):
        - num_workers=2 (was 4 in the original code; 4 reliably triggered
          'received 0 items of ancdata' on 16-GB-shm hosts)
        - pin_memory=None -> auto: True iff CUDA is available
        - prefetch_factor=2 (default)
        - persistent_workers=True (workers survive across epochs, so FDs
          aren't churned every restart)
    The file-descriptor sharing strategy was already switched to
    'file_system' at import time by `_configure_multiprocessing_for_large_volumes`.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=op.join(runs_dir, experiment_tag))

    print(f"CUDA available: {torch.cuda.is_available()}")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # Resolve the dataset root once so we can record it in the report and
    # also reuse it across the three create_dataset() calls without
    # re-running resolution logic.
    resolved_root = resolve_dataset_root(dataset_root)

    # Build the loaders. `persistent_workers=True` is the single most
    # important setting for avoiding the slow FD leak across epochs; only
    # valid when num_workers > 0.
    loader_kwargs = dict(
        batch_size=bs,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_dataset = create_dataset(subset="train", validation_cases=vc, test_cases=tc,
                                   split_mode=split_mode, folds=folds, fold=fold,
                                   seed=seed, dataset_root=resolved_root)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)

    val_dataset = create_dataset(subset="validation", validation_cases=vc, test_cases=tc,
                                   split_mode=split_mode, folds=folds, fold=fold,
                                 seed=seed, dataset_root=resolved_root)
    val_loader = DataLoader(val_dataset, **loader_kwargs)

    # split_mode="final" trains the deployable model on every participant not
    # needed for checkpoint selection. There is deliberately NO test set: the
    # CV run already produced the generalization estimate, and evaluating this
    # model on any subset of its own training data would be circular. Report
    # the CV mean +/- SD as this model's expected performance.
    if split_mode == "final":
        test_dataset = test_loader = None
        print("[split] FINAL MODEL run: no test set will be built or evaluated. "
              "Report the CV mean as this model's performance, not a number "
              "from this run.")
    else:
        test_dataset = create_dataset(subset="test", validation_cases=vc, test_cases=tc,
                                       split_mode=split_mode, folds=folds, fold=fold,
                                      seed=seed, dataset_root=resolved_root)
        test_loader = DataLoader(test_dataset, **loader_kwargs)

    # ---- split verification + manifest ------------------------------------
    # Fail before training rather than after 50k iterations on a leaking split.
    _sub = SPLIT_MANIFEST.get("subjects", {})
    _tr, _va, _te = (set(_sub.get(k, [])) for k in ("train", "validation", "test"))
    _ov = {"train&validation": _tr & _va, "train&test": _tr & _te,
           "validation&test": _va & _te}
    if split_mode == "final":
        print(f"[split] participants -> train {len(_tr)}, val {len(_va)} (no test set)")
    else:
        print(f"[split] participants -> train {len(_tr)}, val {len(_va)}, test {len(_te)}")
        print(f"[split] test participants: {sorted(_te)}")
    _leakage_free = split_mode in ("subject", "final") or (folds and folds >= 2)
    if _leakage_free:
        for _name, _o in _ov.items():
            assert not _o, f"SUBJECT LEAKAGE between {_name}: {sorted(_o)}"
        print("[split] verified: no participant appears in more than one split.")
    else:
        print(f"[split] WARNING: legacy image-wise split; overlaps: "
              f"{ {k: sorted(v) for k, v in _ov.items() if v} }")

    SPLIT_MANIFEST["config"] = {
        "experiment_tag": experiment_tag, "split_mode": split_mode,
        "folds": folds, "fold": fold, "seed": seed,
        "leakage_free": bool(_leakage_free),
    }
    with open(op.join(results_dir, f"{experiment_tag}_split_manifest.json"), "w") as _fh:
        json.dump(SPLIT_MANIFEST, _fh, indent=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = build_model().to(device)
    total_params, trainable_params = count_parameters(net)
    print(f"[{experiment_tag}] total params: {total_params:,}  trainable: {trainable_params:,}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    include_background = False

    # Optimizer selection.
    #
    # `paper`  : torch.optim.Adam(net.parameters()) with default lr=1e-3,
    #            betas=(0.9, 0.999), no weight decay.  This is the recipe
    #            used in model.py — the one that actually produced the
    #            published numbers. It is the DEFAULT here because the
    #            old Adamax(lr=5e-3) recipe, while it happened to work for
    #            the small (c_reduced) variant by luck, destabilises the
    #            full-sized AttentionUnet — the very thing the manuscript
    #            wants to evaluate.
    #
    # `adamax`           : Adamax(lr=lr, betas=(0.95, 0.99), weight_decay=1e-5)
    #                      Kept for backwards-compat with earlier (a) and
    #                      (c_reduced) runs.
    #
    # `adam_paperbc3d`   : torch.optim.Adam(net.parameters(), lr=1e-2,
    #                      weight_decay=1e-4)  -- the recipe used by
    #                      model_bc_3d.py (the published AttentionUnet run
    #                      that performs well on this dataset). Use this
    #                      together with --loss diceloss when re-running
    #                      that exact recipe under the ablation pipeline.
    #
    # Selected via `optimizer_name`; defaults to "paper".
    if optimizer_name == "paper":
        # NB: --lr is IGNORED in the paper recipe to stay faithful to
        # model.py, which uses Adam's defaults verbatim. Pass
        # `--optimizer adamax --lr <whatever>` to override.
        opt = torch.optim.Adam(net.parameters())
        opt_label = "Adam (paper default: lr=1e-3, betas=(0.9, 0.999), wd=0)"
    elif optimizer_name == "adamax":
        opt = torch.optim.Adamax(net.parameters(), lr=lr, betas=(0.95, 0.99),
                                 eps=1e-08, weight_decay=1e-5)
        opt_label = f"Adamax (lr={lr}, betas=(0.95, 0.99), wd=1e-5)"
    elif optimizer_name == "adam_paperbc3d":
        # Exact recipe from model_bc_3d.py. --lr defaults to 5e-3 via the
        # common parser; pass --lr 1e-2 to faithfully reproduce the
        # published bc3d run. --weight-decay is not exposed at the CLI
        # level (nothing else uses it) -- if you need to sweep it, edit
        # the value here.
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        opt_label = f"Adam (paper bc3d: lr={lr}, wd=1e-4)"
    else:
        raise ValueError(
            f"Unknown optimizer_name='{optimizer_name}'. "
            "Use 'paper', 'adamax', or 'adam_paperbc3d'."
        )
    print(f"[run_experiment] optimizer: {opt_label}")

    # Loss selection.
    #
    # `dicefocal` (default): DiceFocalLoss used by the ablation pipeline
    #                        (lambda_dice=0.3, lambda_focal=0.7, gamma=2.5).
    #                        This is the loss under which (a), (b), (c) etc.
    #                        were trained, and is the right choice when you
    #                        want results comparable to those ablations.
    #
    # `diceloss`           : plain DiceLoss(to_onehot_y=True, softmax=True) --
    #                        the exact loss used by model_bc_3d.py. Use this
    #                        together with --optimizer adam_paperbc3d to
    #                        reproduce the bc3d recipe under this pipeline.
    if loss_name == "dicefocal":
        loss_function = DiceFocalLoss(
            include_background=include_background,
            lambda_dice=0.3, lambda_focal=0.7,
            to_onehot_y=True, softmax=True,
            gamma=2.5, smooth_nr=1e-6, smooth_dr=1e-6,
        )
        loss_label = ("DiceFocalLoss (lambda_dice=0.3, lambda_focal=0.7, "
                      "gamma=2.5, include_background=False)")
    elif loss_name == "dicece":
        # DiceCELoss with a heavy foreground weight. Rationale: electrodes
        # occupy a tiny fraction of a 224x288x288 volume, so "predict all
        # background" is a strong local minimum -- exactly the collapse seen
        # in 7 of 24 folds under DiceFocalLoss (zero-Dice cases, inf HD, and
        # c_reduced f3 which stalled at a flat loss of 0.30 and never
        # recovered). Weighting the foreground class 10x in the CE term keeps
        # a gradient pointing away from that minimum. This mirrors the recipe
        # already adopted in model_bc_3d_improved_1.py.
        # include_background is left at the caller's setting for the Dice term;
        # the CE weight vector covers [background, electrode].
        loss_function = DiceCELoss(
            include_background=include_background,
            lambda_dice=0.5, lambda_ce=0.5,
            to_onehot_y=True, softmax=True,
            # must live on the model's device: torch's CrossEntropyLoss
            # errors if `weight` is on CPU while logits are on CUDA.
            ce_weight=torch.tensor([1.0, 10.0], device=device),
        )
        loss_label = ("DiceCELoss (lambda_dice=0.5, lambda_ce=0.5, "
                      "ce_weight=[1,10], include_background="
                      f"{include_background})")
    elif loss_name == "diceloss":
        # Paper bc3d uses DiceLoss with include_background=True implicitly
        # (DiceMetric in model_bc_3d.py also uses include_background=True).
        # We mirror that here for fidelity.
        loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        loss_label = "DiceLoss(to_onehot_y=True, softmax=True) -- paper bc3d"
    else:
        raise ValueError(
            f"Unknown loss_name='{loss_name}'. "
            "Use 'dicefocal', 'dicece', or 'diceloss'."
        )
    print(f"[run_experiment] loss: {loss_label}")

    _AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": torch.float32}
    if amp_dtype not in _AMP:
        raise ValueError(f"amp_dtype must be one of {sorted(_AMP)}, got {amp_dtype!r}")
    amp_label = amp_dtype
    amp_dtype = _AMP[amp_label]
    # GradScaler is only needed for fp16; bf16/fp32 have adequate range.
    use_scaler = amp_dtype is torch.float16
    print(f"[run_experiment] autocast: {amp_label} "
          f"(loss always fp32) | grad-clip: {grad_clip or 'off'}")

    network = Network(
        net=net,
        scaler=torch.amp.GradScaler("cuda", enabled=use_scaler),
        opt=opt,
        loss_function=loss_function,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dice_metric=DiceMetric(reduction="mean", include_background=include_background, ignore_empty=True),
        hausdorff_metric=HausdorffDistanceMetric(include_background=include_background),
        iou_metric=MeanIoU(include_background=include_background, ignore_empty=True),
        eval_num=eval_num,
        max_iterations=max_iterations,
        root_dir=str(results_dir),
        accumulation_steps=accumulation_steps,
        writer=writer,
        ckpt_name=f"{experiment_tag}_best_metric_model.pth",
        debug_tag=experiment_tag,
        min_dice_threshold=min_dice_threshold,
        sanity_probation_iters=sanity_probation_iters,
        max_no_improve_evals=max_no_improve_evals,
    )
    network.amp_dtype = amp_dtype
    network.grad_clip = grad_clip

    torch.cuda.empty_cache()
    while network.global_step < network.max_iterations:
        network.train()
        if network.stop_training:
            print(f"[run_experiment] training stopped early by sanity check "
                  f"at iter {network.global_step}. Reason: {network.failure_reason}")
            break

    print(f"Training done — best val Dice {network.dice_val_best:.4f} @ iter {network.global_step_best}")

    # If the run failed and no checkpoint was ever saved, write a FAILED
    # marker into the results directory and skip the test phase.
    ckpt_path = op.join(network.root_dir, network.ckpt_name)
    run_failed = network.failure_reason is not None or not op.exists(ckpt_path)
    if run_failed:
        failure_reason = (
            network.failure_reason
            or f"No checkpoint at {ckpt_path}: model never reached a "
               f"positive validation Dice."
        )
        print(f"[run_experiment] RUN FAILED: {failure_reason}")
        # Stub out the test metrics so the report writer still produces
        # a valid (clearly-marked) output file the aggregator can read.
        nan = float("nan")
        test_metrics = {
            "dice_mean": nan, "dice_std": nan,
            "hausdorff_mean": nan, "hausdorff_std": nan,
            "iou_mean": nan, "iou_std": nan,
            "dice_values": [], "hausdorff_values": [], "iou_values": [],
            "FAILED": True,
            "failure_reason": failure_reason,
        }
    elif split_mode == "final":
        # No test set exists for a final-model run. Reload the best checkpoint
        # (already the deployable artifact) but skip evaluation -- there is
        # nothing held out to evaluate on, and testing on training/validation
        # data would be circular. Performance is the CV mean from the earlier
        # ablation run, reported alongside this checkpoint, not recomputed here.
        network.net.load_state_dict(torch.load(ckpt_path))
        test_metrics = {
            "dice": None, "hausdorff": None, "iou": None,
            "dice_std": None, "note": "final-model run: no test set; "
                                      "see the CV run for the performance estimate.",
            "FAILED": False,
        }
    else:
        # Reload best checkpoint and evaluate on test set
        network.net.load_state_dict(torch.load(ckpt_path))
        test_metrics = network.test()
        test_metrics["FAILED"] = False

    best_val = {
        "dice": float(network.dice_val_best),
        "hausdorff": float(network.hausdorff_at_best),
        "iou": float(network.iou_at_best),
        "iteration": int(network.global_step_best),
    }

    train_config = {
        "dataset_root": resolved_root,
        "validation_cases": vc,
        "test_cases": (tc if split_mode != "final" else None),
        "is_final_model": split_mode == "final",
        "batch_size": bs,
        "seed": seed,
        "accumulation_steps": accumulation_steps,
        "max_iterations": max_iterations,
        "eval_num": eval_num,
        "optimizer": opt_label,
        "optimizer_name": optimizer_name,
        "learning_rate": lr if optimizer_name == "adamax" else "1e-3 (Adam default; --lr ignored)",
        "loss": loss_label,
        "loss_name": loss_name,
        "scheduler": "CosineAnnealingWarmRestarts (T_0=1000, eta_min=1e-5)",
        "include_background": include_background,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "persistent_workers": num_workers > 0,
        "sharing_strategy": "file_system",
        "min_dice_threshold": min_dice_threshold,
        "sanity_probation_iters": sanity_probation_iters,
        "max_no_improve_evals": max_no_improve_evals,
    }

    write_report(
        experiment_tag=experiment_tag,
        experiment_description=experiment_description,
        model_config=model_config_summary,
        train_config=train_config,
        param_counts=(total_params, trainable_params),
        best_val=best_val,
        test_metrics=test_metrics,
        results_dir=results_dir,
    )

    writer.close()
