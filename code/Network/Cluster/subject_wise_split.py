"""
Subject-wise (group-aware) train/validation/test splitting for the electrode
segmentation pipeline.

WHY THIS MODULE EXISTS
----------------------
The original `subsetting()` in `ablation_utils.py` split the dataset
*image-wise*: it built a label vector of length ``n_cases`` and shuffled it.
Because a single participant contributes several images (multiple sessions x
multiple runs, e.g. ``sub-022`` with ses-1/ses-2 x run-01/run-02 = 4 images),
images of the *same head* could be distributed across train, validation and
test. Those images are near-duplicates: same participant, same neuronavigated
electrode montage, acquired minutes apart and co-registered to a common
baseline. A test image whose sibling run is in the training set therefore does
not measure generalisation to a new participant.

This module replaces that behaviour with a *subject-wise* split: whole
participants are assigned to exactly one of train/validation/test, so no
participant ever appears in more than one split.

WHAT IS PROVIDED
----------------
``subject_wise_subsetting``  drop-in replacement for ``subsetting()``
``subsetting_legacy_imagewise``  the ORIGINAL image-wise behaviour, kept
                                 verbatim so previously published results stay
                                 exactly reproducible
``split_report``             returns the full assignment (which subject went
                             where) without building any dataset
``audit_leakage``            reports which subjects span splits under a given
                             mode -- use this to quantify leakage in an
                             already-completed run *without retraining*

Only numpy is required, so this module can be unit-tested without MONAI/torch.
"""

from __future__ import annotations

import re
import warnings
from collections import OrderedDict

import numpy as np

__all__ = [
    "extract_subject_id",
    "group_indices_by_subject",
    "subject_wise_subsetting",
    "subsetting_legacy_imagewise",
    "split_report",
    "audit_leakage",
    "subject_wise_kfold",
    "kfold_subsetting",
    "final_split_report",
    "final_subsetting",
]

_SUBJECT_RE = re.compile(r"sub-(\d+)")


# --------------------------------------------------------------------------- #
# Subject identification
# --------------------------------------------------------------------------- #
def extract_subject_id(path_or_name):
    """Return the participant identifier, e.g. ``'sub-022'``, or None.

    Deliberately matches ONLY the ``sub-<digits>`` component, ignoring
    ``ses-`` and ``run-``. That is the whole point: the session and run are
    exactly what must NOT separate two images of the same head.

    Works on full paths, bare filenames and mask folder names, because the
    ``sub-XXX`` token appears in all of them:

        /.../sub-022/unzipped/rsub-022_ses-2_acq-petra_run-01_PDw.nii -> sub-022
        sub-022_ses-2_run-01                                          -> sub-022

    Note the regex is applied to the *whole* string, so a path that contains
    the subject folder still resolves correctly even if the filename were
    ever renamed.
    """
    if path_or_name is None:
        return None
    m = _SUBJECT_RE.search(str(path_or_name))
    return f"sub-{m.group(1)}" if m else None


def group_indices_by_subject(vols, strict=True):
    """Map subject id -> list of case indices, preserving input order.

    Parameters
    ----------
    vols : sequence of str
        Volume paths, in the same order as the paired mask list.
    strict : bool
        If True (default), raise when a subject id cannot be parsed. Silently
        dropping such a case would reintroduce exactly the class of quiet bug
        this module exists to remove.

    Returns
    -------
    OrderedDict[str, list[int]]
    """
    groups = OrderedDict()
    unparsed = []
    for i, v in enumerate(vols):
        sid = extract_subject_id(v)
        if sid is None:
            unparsed.append(v)
            continue
        groups.setdefault(sid, []).append(i)

    if unparsed:
        msg = (
            f"Could not extract a 'sub-XXX' id from {len(unparsed)} case(s), "
            f"e.g. {unparsed[:3]}. Subject-wise splitting is impossible for "
            f"these and they would silently risk leakage."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning)
    return groups


# --------------------------------------------------------------------------- #
# The split itself
# --------------------------------------------------------------------------- #
def _greedy_fill(subject_order, sizes, target, remaining):
    """Pick whole subjects from `subject_order` to get as close to `target` cases.

    Greedy with a stop rule: keep adding subjects while doing so moves the
    bucket size CLOSER to the target. Because subjects contribute unequal
    numbers of images, an exact hit is generally impossible -- we accept the
    nearest achievable size rather than splitting a subject, which is the
    whole point.

    Returns (chosen_subjects, remaining_subjects).
    """
    chosen, current = [], 0
    remaining = list(remaining)
    for sid in subject_order:
        if sid not in remaining:
            continue
        k = sizes[sid]
        # Stop if adding this subject overshoots further than staying put.
        if abs(current + k - target) > abs(current - target):
            continue
        chosen.append(sid)
        remaining.remove(sid)
        current += k
        if current >= target:
            break
    return chosen, remaining


def split_report(vols, validation_cases, test_cases, seed=42, strict=True):
    """Compute the subject-wise assignment WITHOUT touching any data.

    Returns a dict with per-split subject lists, per-split case indices, and
    the achieved vs requested sizes. Useful for logging, for the Methods
    section, and for unit tests.
    """
    groups = group_indices_by_subject(vols, strict=strict)
    sizes = {sid: len(ix) for sid, ix in groups.items()}
    n_cases = sum(sizes.values())

    if validation_cases + test_cases >= n_cases:
        raise ValueError(
            f"validation+test ({validation_cases + test_cases}) must be < "
            f"total cases ({n_cases})."
        )
    if len(groups) < 3:
        raise ValueError(
            f"Only {len(groups)} subject(s) found; a subject-wise 3-way split "
            f"needs at least 3."
        )

    # Dedicated RNG: does NOT touch numpy's global random state (the original
    # subsetting() called np.random.seed(), which silently reseeded the global
    # RNG three times per run).
    rng = np.random.default_rng(seed)
    subject_order = sorted(groups)               # deterministic starting point
    rng.shuffle(subject_order)                   # then a seeded permutation

    remaining = list(subject_order)
    test_subjects, remaining = _greedy_fill(subject_order, sizes, test_cases, remaining)
    val_subjects, remaining = _greedy_fill(subject_order, sizes, validation_cases, remaining)
    train_subjects = [s for s in subject_order if s in remaining]

    for name, subs in (("test", test_subjects), ("validation", val_subjects),
                       ("train", train_subjects)):
        if not subs:
            raise ValueError(
                f"Subject-wise split produced an empty '{name}' set. With "
                f"{len(groups)} subjects and targets "
                f"val={validation_cases}/test={test_cases}, the requested "
                f"sizes are not achievable without splitting a subject."
            )

    def _indices(subs):
        return sorted(i for s in subs for i in groups[s])

    report = {
        "n_cases": n_cases,
        "n_subjects": len(groups),
        "subjects": {"train": train_subjects, "validation": val_subjects,
                     "test": test_subjects},
        "indices": {"train": _indices(train_subjects),
                    "validation": _indices(val_subjects),
                    "test": _indices(test_subjects)},
        "requested": {"validation": validation_cases, "test": test_cases,
                      "train": n_cases - validation_cases - test_cases},
        "seed": seed,
    }
    report["achieved"] = {k: len(v) for k, v in report["indices"].items()}

    # Hard guarantee: the invariant this module exists to enforce.
    all_sets = [set(report["subjects"][k]) for k in ("train", "validation", "test")]
    for a in range(3):
        for b in range(a + 1, 3):
            overlap = all_sets[a] & all_sets[b]
            assert not overlap, f"subject leakage: {sorted(overlap)}"
    assert sum(report["achieved"].values()) == n_cases, "cases lost in split"
    return report


def subject_wise_subsetting(subset, vols, mask, validation_cases, test_cases,
                            seed=42, strict=True, verbose=True):
    """Drop-in replacement for ``subsetting()`` with NO subject leakage.

    Same signature and same return type ``(vols_subset, mask_subset)`` as the
    original, so ``create_dataset`` needs a one-line change.

    Unlike the original, the requested validation/test sizes are TARGETS, not
    guarantees: whole subjects are allocated, so the achieved sizes are the
    nearest reachable ones. The achieved sizes are printed (and available via
    ``split_report``) so they can be reported accurately in a manuscript.
    """
    vols = np.asarray(vols)
    mask = np.asarray(mask)
    if len(vols) != len(mask):
        raise ValueError(f"vols/mask length mismatch: {len(vols)} vs {len(mask)}")
    if subset not in ("train", "validation", "test"):
        raise ValueError(f"unknown subset '{subset}'")

    rep = split_report(vols, validation_cases, test_cases, seed=seed, strict=strict)

    if verbose:
        a, r = rep["achieved"], rep["requested"]
        print(f"[split] subject-wise, seed={seed}: "
              f"{rep['n_cases']} cases / {rep['n_subjects']} subjects -> "
              f"train {a['train']} (req {r['train']}), "
              f"val {a['validation']} (req {r['validation']}), "
              f"test {a['test']} (req {r['test']})")
        print(f"[split]   test subjects: {rep['subjects']['test']}")
        print(f"[split]   val  subjects: {rep['subjects']['validation']}")
        print(f"[split]   NO subject appears in more than one split.")

    idx = rep["indices"][subset]
    return vols[idx], mask[idx]


# --------------------------------------------------------------------------- #
# Legacy behaviour, preserved verbatim for reproducibility
# --------------------------------------------------------------------------- #
def subsetting_legacy_imagewise(subset, vols, mask, validation_cases,
                                test_cases, seed=42, warn=True):
    """The ORIGINAL image-wise split. Reproduces previously published runs.

    Kept because published results must remain reproducible. It is NOT
    leakage-free and must not be used for new results.
    """
    if warn:
        warnings.warn(
            "Using the legacy IMAGE-WISE split: images of the same participant "
            "can land in different splits, so test metrics are optimistic. "
            "Use subject_wise_subsetting() for new results.",
            RuntimeWarning, stacklevel=2,
        )
    vols = np.asarray(vols)
    mask = np.asarray(mask)
    np.random.seed(seed)
    train_cases = len(vols) - (validation_cases + test_cases)
    sampling_array = np.hstack([
        np.repeat("train", train_cases),
        np.repeat("validation", validation_cases),
        np.repeat("test", test_cases),
    ])
    np.random.shuffle(sampling_array)
    return vols[sampling_array == subset], mask[sampling_array == subset]


# --------------------------------------------------------------------------- #
# Leakage audit -- quantify an EXISTING run without retraining
# --------------------------------------------------------------------------- #
def audit_leakage(vols, validation_cases, test_cases, seed=42, mode="legacy"):
    """Report which subjects span splits, and which test cases are contaminated.

    Run this with ``mode='legacy'`` and the seed used for a completed training
    run to find out, retrospectively, exactly which test images had a sibling
    image of the same participant in the training set. Those cases can then be
    excluded from the reported test metrics (using saved per-case metrics), so
    a leakage-free estimate can be obtained WITHOUT retraining.

    Returns a dict with:
      ``crossing_subjects``     subjects appearing in >1 split
      ``clean_test_indices``    test cases whose subject is absent from train+val
      ``contaminated_test_indices``
    """
    vols = np.asarray(vols)
    if mode == "legacy":
        assign = {}
        np.random.seed(seed)
        train_cases = len(vols) - (validation_cases + test_cases)
        arr = np.hstack([
            np.repeat("train", train_cases),
            np.repeat("validation", validation_cases),
            np.repeat("test", test_cases),
        ])
        np.random.shuffle(arr)
        for i, lab in enumerate(arr):
            assign[i] = lab
    elif mode == "subject":
        rep = split_report(vols, validation_cases, test_cases, seed=seed)
        assign = {}
        for lab, idxs in rep["indices"].items():
            for i in idxs:
                assign[i] = lab
    else:
        raise ValueError("mode must be 'legacy' or 'subject'")

    by_subject = {}
    for i, v in enumerate(vols):
        by_subject.setdefault(extract_subject_id(v), []).append(i)

    crossing = {sid: sorted({assign[i] for i in idxs})
                for sid, idxs in by_subject.items()
                if len({assign[i] for i in idxs}) > 1}

    train_val_subjects = {sid for sid, idxs in by_subject.items()
                          if any(assign[i] in ("train", "validation") for i in idxs)}

    test_idx = [i for i, lab in assign.items() if lab == "test"]
    clean, dirty = [], []
    for i in test_idx:
        (dirty if extract_subject_id(vols[i]) in train_val_subjects else clean).append(i)

    return {
        "mode": mode,
        "seed": seed,
        "n_cases": len(vols),
        "n_subjects": len(by_subject),
        "crossing_subjects": crossing,
        "n_crossing_subjects": len(crossing),
        "test_indices": sorted(test_idx),
        "clean_test_indices": sorted(clean),
        "contaminated_test_indices": sorted(dirty),
        "clean_test_files": [str(vols[i]) for i in sorted(clean)],
        "contaminated_test_files": [str(vols[i]) for i in sorted(dirty)],
    }


# --------------------------------------------------------------------------- #
# Subject-wise k-fold cross-validation
# --------------------------------------------------------------------------- #
def subject_wise_kfold(vols, n_folds=5, seed=42, strict=True):
    """Partition PARTICIPANTS into ``n_folds`` groups of near-equal image count.

    Every image is used as test exactly once across the folds, which is what
    makes a k-fold ablation table far more stable than a single small test
    split. Participants -- not images -- are the unit, so there is no leakage.

    Balancing: participants contribute unequal numbers of images (1-4 here), so
    folds are filled greedily largest-first into whichever fold currently holds
    the fewest images. This keeps fold sizes close (e.g. 11/11/11/11/10 on 54
    images from 38 participants) without ever splitting a participant.

    Returns list[list[str]] -- the participant ids per fold.
    """
    groups = group_indices_by_subject(vols, strict=strict)
    sizes = {sid: len(ix) for sid, ix in groups.items()}
    if len(groups) < n_folds:
        raise ValueError(
            f"{len(groups)} participants cannot fill {n_folds} folds."
        )

    rng = np.random.default_rng(seed)
    order = sorted(groups)
    rng.shuffle(order)
    # Largest participants first -> better balance; ties broken by the seeded
    # shuffle above, so the assignment is random but reproducible.
    order.sort(key=lambda s: -sizes[s])

    folds = [[] for _ in range(n_folds)]
    counts = [0] * n_folds
    for sid in order:
        i = counts.index(min(counts))
        folds[i].append(sid)
        counts[i] += sizes[sid]
    return [sorted(f) for f in folds]


def kfold_subsetting(subset, vols, mask, fold, n_folds=5, seed=42,
                     strict=True, verbose=True):
    """Subject-wise k-fold split for one fold. Drop-in for ``subsetting()``.

    For fold *k* (0-indexed):
        test       = participants of fold k
        validation = participants of fold (k+1) % n_folds
        train      = all remaining participants

    Rotating the validation fold means no extra hyper-parameter and no
    additional held-out set: across the k runs every participant serves as
    test exactly once and as validation exactly once.
    """
    vols = np.asarray(vols)
    mask = np.asarray(mask)
    if len(vols) != len(mask):
        raise ValueError(f"vols/mask length mismatch: {len(vols)} vs {len(mask)}")
    if subset not in ("train", "validation", "test"):
        raise ValueError(f"unknown subset '{subset}'")
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")

    folds = subject_wise_kfold(vols, n_folds=n_folds, seed=seed, strict=strict)
    groups = group_indices_by_subject(vols, strict=strict)

    test_subj = set(folds[fold])
    val_subj = set(folds[(fold + 1) % n_folds])
    train_subj = set(groups) - test_subj - val_subj

    chosen = {"train": train_subj, "validation": val_subj, "test": test_subj}
    for name, subs in chosen.items():
        if not subs:
            raise ValueError(f"fold {fold} produced an empty '{name}' set.")
    # Invariant check.
    assert not (test_subj & val_subj), "fold overlap: test/validation"
    assert not (train_subj & test_subj), "fold overlap: train/test"
    assert not (train_subj & val_subj), "fold overlap: train/validation"

    if verbose:
        n = {k: sum(len(groups[s]) for s in v) for k, v in chosen.items()}
        print(f"[split] subject-wise {n_folds}-fold CV, fold {fold}, seed={seed}: "
              f"{len(vols)} cases / {len(groups)} participants -> "
              f"train {n['train']}, val {n['validation']}, test {n['test']}")
        print(f"[split]   test participants ({len(test_subj)}): {sorted(test_subj)}")
        print(f"[split]   val  participants ({len(val_subj)}): {sorted(val_subj)}")
        print("[split]   NO participant appears in more than one split.")

    idx = sorted(i for s in chosen[subset] for i in groups[s])
    return vols[idx], mask[idx]


# --------------------------------------------------------------------------- #
# Final-model training: 2-way subject-wise split, NO test set
# --------------------------------------------------------------------------- #
def final_split_report(vols, validation_cases, seed=42, strict=True):
    """Whole-participant TRAIN/VALIDATION split with no test set at all.

    Use this to train the deployable model after CV has already told you
    which architecture and roughly what performance to expect: the CV mean is
    your generalization estimate, so this final run can use every image that
    isn't needed to pick a checkpoint. validation_cases is a TARGET (see
    split_report); the achieved size is returned and should be reported.
    """
    groups = group_indices_by_subject(vols, strict=strict)
    sizes = {sid: len(ix) for sid, ix in groups.items()}
    n_cases = sum(sizes.values())
    if validation_cases >= n_cases:
        raise ValueError(f"validation_cases ({validation_cases}) must be < total ({n_cases}).")
    if len(groups) < 2:
        raise ValueError(f"Only {len(groups)} participant(s); need at least 2.")

    rng = np.random.default_rng(seed)
    order = sorted(groups)
    rng.shuffle(order)

    val_subjects, remaining = _greedy_fill(order, sizes, validation_cases, list(order))
    train_subjects = [s for s in order if s in remaining]
    if not val_subjects or not train_subjects:
        raise ValueError("final split produced an empty train or validation set.")

    def _idx(subs):
        return sorted(i for s in subs for i in groups[s])

    report = {
        "n_cases": n_cases, "n_subjects": len(groups),
        "subjects": {"train": train_subjects, "validation": val_subjects},
        "indices": {"train": _idx(train_subjects), "validation": _idx(val_subjects)},
        "requested": {"validation": validation_cases,
                      "train": n_cases - validation_cases},
        "seed": seed,
    }
    report["achieved"] = {k: len(v) for k, v in report["indices"].items()}
    assert not (set(train_subjects) & set(val_subjects)), "leakage in final split"
    assert sum(report["achieved"].values()) == n_cases
    return report


def final_subsetting(subset, vols, mask, validation_cases, seed=42,
                     strict=True, verbose=True):
    """2-way subject-wise split for final-model training. subset: 'train'|'validation'.

    There is deliberately no 'test' option: by the time you train the final
    model, the CV run already produced the generalization estimate. Training
    a final model on 100% of participants (minus a small validation slice
    kept only for checkpoint selection) maximizes the data behind the model
    you actually ship, and reusing the CV number as its reported performance
    avoids the circularity of testing on data the final model was trained on.
    """
    vols = np.asarray(vols); mask = np.asarray(mask)
    if len(vols) != len(mask):
        raise ValueError(f"vols/mask length mismatch: {len(vols)} vs {len(mask)}")
    if subset not in ("train", "validation"):
        raise ValueError(f"final_subsetting has no '{subset}' subset (only train/validation)")

    rep = final_split_report(vols, validation_cases, seed=seed, strict=strict)
    if verbose:
        a, r = rep["achieved"], rep["requested"]
        print(f"[split] FINAL MODEL, subject-wise, seed={seed}: "
              f"{rep['n_cases']} cases / {rep['n_subjects']} subjects -> "
              f"train {a['train']} (req {r['train']}), "
              f"validation {a['validation']} (req {r['validation']})")
        print(f"[split]   validation subjects: {rep['subjects']['validation']}")
        print("[split]   NO test set -- report the CV mean as this model's "
              "expected performance, not a number from this run.")
    idx = rep["indices"][subset]
    return vols[idx], mask[idx]
