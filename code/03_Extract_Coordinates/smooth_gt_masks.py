#!/usr/bin/env python3
"""
==============================================================
  smooth_gt_masks.py
==============================================================
  Re-smooths the expert electrode masks of dataset_RU.

  THE DEFECT
  ----------
  The smoothing originally applied left a thin empty layer,
  usually one voxel, between the surface of an electrode and its
  body. Connected-component labelling therefore splits a single
  electrode into two or more pieces: in the 72 masks of this
  dataset, 35 contain at least one electrode represented by more
  than one component, and only 67 yield exactly four after
  bridging. Downstream this inflates cluster counts, corrupts
  per-electrode volumes and, because these masks were used for
  training, teaches the networks to reproduce the artefact.

  WHAT "CORRECTLY" MEANS HERE
  ---------------------------
  Smoothing that removes the artefact must satisfy four things,
  and this script checks all four rather than assuming them:

    1. each electrode ends as ONE connected component;
    2. electrodes are never merged with each other;
    3. the change in volume is small and is reported, not
       assumed --- closing necessarily adds voxels, and a method
       that claims otherwise is not measuring;
    4. the outer boundary is not inflated. Closing is applied
       per electrode and only where it changes connectivity, so
       a smooth, already-connected surface is left alone.

  Operations are applied PER ELECTRODE, never to the whole mask
  at once. A global closing with a kernel large enough to bridge
  a gap would also bridge the space between two adjacent
  electrodes, which is the failure mode the whole pipeline is
  trying to avoid.

  ORIGINALS ARE NEVER MODIFIED.

--------------------------------------------------------------
  HOW TO USE
      python smooth_gt_masks.py --root <dataset_RU path> --dry-run
      python smooth_gt_masks.py --root <dataset_RU path>
      python smooth_gt_masks.py --root <...> --radius 2 --median
==============================================================
"""

import os
import glob
import argparse
import csv as _csv

import numpy as np
import nibabel as nib
from scipy.ndimage import (label, binary_dilation, binary_closing,
                           binary_fill_holes, median_filter,
                           generate_binary_structure, iterate_structure)


def ball(radius):
    """Spherical structuring element of the given radius in voxels."""
    if radius <= 1:
        return generate_binary_structure(3, 1)
    return iterate_structure(generate_binary_structure(3, 1), int(radius))


def group_electrodes(binary, bridge):
    """Label electrodes, treating fragments within `bridge` voxels as one.

    Dilation decides the GROUPING only; the labels are painted back onto the
    original voxels, so this step adds nothing.
    """
    if bridge <= 0:
        return label(binary)
    grown, _ = label(binary_dilation(binary, ball(bridge)))
    painted = np.where(binary, grown, 0)
    ids = [v for v in np.unique(painted) if v != 0]
    out = np.zeros_like(painted)
    for new_id, old in enumerate(ids, start=1):
        out[painted == old] = new_id
    return out, len(ids)


def smooth_one_electrode(mask, radius, do_median, fill_holes,
                         speck_min=8):
    """Smooth a single electrode. Returns the result and what was done."""
    steps = []
    out = mask.copy()

    if do_median:
        # Speckle is removed by DELETING small isolated components, not by a
        # median filter. A median erodes a thin surface disc -- in testing it
        # removed 9% of the annotated volume and damaged masks that were
        # already correct -- because the disc is one voxel thick and a 3-voxel
        # median has no majority to preserve it. Dropping components below a
        # size threshold removes exactly the speckle and touches nothing else.
        lab, n = label(out)
        if n > 1:
            sizes = np.array([0] + [(lab == i).sum() for i in range(1, n + 1)])
            small = np.isin(lab, np.where(sizes < speck_min)[0])
            small &= lab > 0
            if small.any() and (out.sum() - small.sum()) > 0:
                steps.append(f"dropped {int(small.sum())} speckle voxel(s)")
                out = out & ~small

    if label(out)[1] > 1:
        # Closing bridges the empty layer. It is applied only when the
        # electrode is actually split; a connected electrode is left untouched,
        # so smoothing never inflates an already-correct surface.
        closed = binary_closing(out, ball(radius))
        if label(closed)[1] < label(out)[1]:
            steps.append(f"closing r={radius}")
            out = closed

    if fill_holes:
        filled = binary_fill_holes(out)
        if filled.sum() != out.sum():
            steps.append("fill-holes")
            out = filled

    return out, steps


def process(path, a):
    img = nib.load(path)
    binary = np.asanyarray(img.dataobj) > 0
    vox_mm3 = float(abs(np.linalg.det(np.asarray(img.affine)[:3, :3])))

    n_before = label(binary)[1]

    # Speckle must be removed BEFORE grouping. A stray voxel lying within the
    # grouping distance of an electrode is absorbed into that electrode's group
    # and is then no longer isolated, so it can never be identified as speckle
    # afterwards.
    speck_removed = 0
    if a.median:
        lab, n = label(binary)
        if n > a.expect:
            sizes = np.array([0] + [(lab == i).sum() for i in range(1, n + 1)])
            small = np.isin(lab, np.where(sizes < a.speckle_min)[0]) & (lab > 0)
            if small.any():
                speck_removed = int(small.sum())
                binary = binary & ~small

    grouped, n_groups = group_electrodes(binary, a.bridge)
    # If grouping produced fewer groups than electrodes, two electrodes were
    # merged by the dilation and smoothing them as one would fuse them
    # permanently. Refuse rather than proceed.
    if n_groups < a.expect:
        return dict(
            tag=os.path.basename(os.path.dirname(path)), path=path,
            components_before=n_before, groups=n_groups,
            components_after=n_before, voxels_before=int(binary.sum()),
            voxels_added=0, pct_added=0.0,
            volume_before_mm3=0.0, volume_after_mm3=0.0,
            steps=f"REFUSED: grouping gave {n_groups} groups for "
                  f"{a.expect} expected electrodes; lower --bridge",
            ok=False), binary, img

    out = np.zeros_like(binary)
    all_steps = []
    for gid in range(1, n_groups + 1):
        g = grouped == gid
        sm, steps = smooth_one_electrode(g, a.radius, a.median,
                                         a.fill_holes, a.speckle_min)
        # Guard: an electrode must not grow into a neighbour. If it would, the
        # smoothing is rejected for that electrode and the original kept.
        others = (grouped > 0) & ~g
        if (sm & others).any() or (binary_dilation(sm, ball(1)) & others).any():
            out |= g
            all_steps.append("rejected: would touch a neighbour")
            continue
        out |= sm
        all_steps += steps

    if speck_removed:
        all_steps.append(f"dropped {speck_removed} speckle voxel(s)")
    n_after = label(out)[1]
    added = int(out.sum()) - int(binary.sum())
    return dict(
        tag=os.path.basename(os.path.dirname(path)), path=path,
        components_before=n_before, groups=n_groups, components_after=n_after,
        voxels_before=int(binary.sum()), voxels_added=added,
        pct_added=round(added / max(int(binary.sum()), 1) * 100, 4),
        volume_before_mm3=round(int(binary.sum()) * vox_mm3, 1),
        volume_after_mm3=round(int(out.sum()) * vox_mm3, 1),
        steps="; ".join(sorted(set(all_steps))) or "none",
        ok=(n_after == a.expect),
    ), out, img


def main(a):
    root = os.path.expanduser(a.root)
    masks = sorted(glob.glob(os.path.join(root, "sub-*", "unzipped", "*",
                                          "mask.nii.gz")))
    if not masks:
        masks = sorted(glob.glob(os.path.join(root, "**", "sub-*", "unzipped",
                                              "*", "mask.nii.gz"),
                                 recursive=True))
    if not masks:
        raise SystemExit(
            f"No mask.nii.gz under {root}\n"
            f"Expected <root>/sub-*/unzipped/<run>/mask.nii.gz\n"
            f"Check with: find {root} -name mask.nii.gz | head")

    print(f"\n{'=' * 68}\n  Smooth ground-truth electrode masks\n{'=' * 68}")
    print(f"  Root      : {root}")
    print(f"  Masks     : {len(masks)}")
    print(f"  Expecting : {a.expect} electrodes per mask")
    print(f"  Closing   : radius {a.radius} voxel(s), applied per electrode "
          f"and only where it changes connectivity")
    print(f"  Speckle   : {"remove <" + str(a.speckle_min) + " vox" if a.median else "off"}"
          f"     Fill holes: {'on' if a.fill_holes else 'off'}")
    print(f"  Mode      : {'DRY RUN' if a.dry_run else 'writing ' + a.out_name}")
    print(f"{'=' * 68}\n")

    rows, fixed, already, failed = [], 0, 0, 0
    tot_before = tot_added = 0
    for p in masks:
        try:
            rec, out, img = process(p, a)
        except Exception as exc:                                # noqa: BLE001
            print(f"  UNREADABLE {os.path.basename(os.path.dirname(p))}: "
                  f"{type(exc).__name__}: {exc}")
            failed += 1
            continue
        rows.append(rec)
        tot_before += rec["voxels_before"]
        tot_added += rec["voxels_added"]

        if rec["components_before"] == a.expect and rec["voxels_added"] == 0:
            already += 1
        elif rec["ok"]:
            fixed += 1
            print(f"  {rec['tag']:<30} {rec['components_before']:>3} -> "
                  f"{rec['components_after']:<3} +{rec['voxels_added']:>5} vox "
                  f"({rec['pct_added']:5.2f}%)  [{rec['steps']}]")
        else:
            failed += 1
            print(f"  {rec['tag']:<30} {rec['components_before']:>3} -> "
                  f"{rec['components_after']:<3}  NOT RESOLVED  [{rec['steps']}]")

        if not a.dry_run and rec["ok"] and rec["voxels_added"] != 0:
            dst = (os.path.join(os.path.expanduser(a.out_root),
                                os.path.relpath(p, root)) if a.out_root
                   else os.path.join(os.path.dirname(p), a.out_name))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            nib.save(nib.Nifti1Image(out.astype(np.uint8), img.affine,
                                     img.header), dst)

    print(f"\n{'=' * 68}")
    print(f"  already correct   : {already}")
    print(f"  smoothed to {a.expect}     : {fixed}")
    print(f"  NOT resolved      : {failed}")
    pct = tot_added / max(tot_before, 1) * 100
    print(f"  voxels added      : {tot_added} of {tot_before} ({pct:.3f}%)")
    if pct > 2:
        print("  WARNING: more than 2% of annotated volume added. That is not a")
        print("           thin gap. Inspect the largest contributors before use.")
    print(f"{'=' * 68}\n")
    print("  Smoothing cannot bridge a gap without adding voxels; the figure")
    print("  above is that cost, measured rather than assumed. Report it if the")
    print("  smoothed masks are used for anything published.\n")

    if a.report and rows:
        with open(a.report, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  Report: {a.report}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-root", dest="out_root", default=None,
                    help="mirror the tree here instead of writing beside the "
                         "originals")
    ap.add_argument("--out-name", dest="out_name", default="mask_smoothed.nii.gz")
    ap.add_argument("--radius", type=int, default=1,
                    help="closing radius in voxels; 1 bridges a one-voxel gap")
    ap.add_argument("--bridge", type=int, default=1,
                    help="dilation used only to group fragments of ONE "
                         "electrode before smoothing; adds nothing. Keep it "
                         "small: a value large enough to reach a neighbouring "
                         "electrode would group two electrodes as one and the "
                         "smoothing would then merge them.")
    ap.add_argument("--speckle-min", dest="speckle_min", type=int, default=8,
                    help="with --median, isolated components smaller than this "
                         "many voxels are deleted as annotation speckle")
    ap.add_argument("--expect", type=int, default=4)
    ap.add_argument("--median", action="store_true",
                    help="apply a 3-voxel median filter first, to remove "
                         "annotation speckle")
    ap.add_argument("--fill-holes", dest="fill_holes", action="store_true",
                    help="fill fully enclosed cavities inside an electrode")
    ap.add_argument("--report", default="smoothing_report.csv")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    main(ap.parse_args())
