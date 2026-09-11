#!/usr/bin/env python3
"""
==============================================================
  08_compare_volumes.py
==============================================================
  Compares the segmented electrode volumes of every network
  against the expert ground-truth masks.

  WHY VOLUME
  ----------
  The four electrodes of the montage are physically identical
  objects, so their segmented volume is an ABSOLUTE reference:
  unlike Dice it needs no per-image annotation, and can therefore
  be evaluated on the whole corpus rather than only the annotated
  subset. It answers two questions overlap cannot:

    bias        does a network systematically over- or
                under-segment relative to the expert masks?
    consistency are the four electrodes of one image segmented to
                similar size? The ground truth's own spread is
                the floor here -- a network cannot be expected to
                beat the annotation it was trained on.

  Volume is blind to position: a network could get every volume
  right and place the electrodes badly. It is therefore a
  complement to the extraction-success and coordinate-agreement
  results, not a substitute.

  TWO COMPARISONS
    global  every network's volume distribution against the
            ground-truth reference. Uses all images.
    paired  the same images, network against expert mask, one
            electrode at a time. Restricted to the annotated
            subset, but it removes between-image variation and is
            the stronger test.

--------------------------------------------------------------
  HOW TO USE
      python 08_compare_volumes.py --networks a_baseline b_no_attention \\
             c_reduced d_increased proposed --gt-root /path/to/dataset_RU/...
      python 08_compare_volumes.py --networks proposed          # global only
==============================================================
"""

import os
import re
import glob
import argparse
import statistics as st
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import (label, binary_dilation, binary_closing,
                           binary_fill_holes, generate_binary_structure,
                           iterate_structure)


# ---------------------------------------------------------------------------
# Optional re-smoothing of the expert masks
#
# WHY THIS IS OPT-IN, AND USUALLY SHOULD STAY OFF.
# The masks contain a thin empty layer, usually one voxel, left by the original
# smoothing, which splits an electrode into several connected components. For
# MEASURING volume that defect is fully handled by gap bridging: the dilation
# decides only how fragments are grouped, the voxels are counted on the
# original mask, and nothing is added. Smoothing, by contrast, must add voxels
# to close the gap, and those voxels go straight into the reference volume that
# every network is then compared against. Bridging is therefore the better
# choice for measurement, and smoothing is the right operation only when the
# masks themselves are the product -- for retraining, for instance.
#
# --smooth is provided so the two can be compared, and it reports the reference
# volume both before and after so the cost is visible rather than absorbed.
# ---------------------------------------------------------------------------

def _ball(radius):
    if radius <= 1:
        return generate_binary_structure(3, 1)
    return iterate_structure(generate_binary_structure(3, 1), int(radius))


def smooth_mask(binary, expect=4, radius=1, bridge=1, speckle_min=8,
                fill_holes=False):
    """Close intra-electrode gaps per electrode. Returns (mask, voxels_added).

    Returns the mask unchanged if the operation cannot be performed safely:
    if grouping yields fewer groups than expected electrodes, two electrodes
    were merged by the grouping dilation and smoothing them as one would fuse
    them permanently.
    """
    before = int(binary.sum())

    lab, n = label(binary)
    if n > expect:                      # drop isolated speckle first
        sizes = np.array([0] + [(lab == i).sum() for i in range(1, n + 1)])
        small = np.isin(lab, np.where(sizes < speckle_min)[0]) & (lab > 0)
        binary = binary & ~small

    grown, _ = label(binary_dilation(binary, _ball(bridge)))
    painted = np.where(binary, grown, 0)
    ids = [v for v in np.unique(painted) if v != 0]
    if len(ids) < expect:
        return binary, int(binary.sum()) - before

    out = np.zeros_like(binary)
    for gid in ids:
        g = painted == gid
        s = g
        if label(g)[1] > 1:
            c = binary_closing(g, _ball(radius))
            if label(c)[1] < label(g)[1]:
                s = c
        if fill_holes:
            s = binary_fill_holes(s)
        others = (painted > 0) & ~g
        if (binary_dilation(s, _ball(1)) & others).any():
            s = g                       # would touch a neighbour: keep as is
        out |= s
    return out, int(out.sum()) - before

ELECTRODES = ["anode", "cathode1", "cathode2", "cathode3"]
VOL_COLS = [f"vol_{e}_mm3" for e in ELECTRODES]


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def _axis_bridge(binary, axis, iters=1):
    """Dilate along ONE axis only.

    The gap left inside an electrode runs through its thickness; the space
    separating two electrodes lies in the plane of the scalp. An isotropic
    dilation cannot tell them apart: large enough to close the gap, it also
    reaches sideways and fuses neighbouring electrodes, which is why an
    isotropic bridge gives too few components and no bridge at all gives far
    too many. Dilating along the thin axis alone closes the gap without
    reducing the in-plane separation.
    """
    st = np.zeros((3, 3, 3), dtype=bool)
    sl = [1, 1, 1]
    for k in (0, 1, 2):
        sl[axis] = k
        st[tuple(sl)] = True
    return binary_dilation(binary, st, iterations=iters)


def _label_electrodes(binary, expect, bridge, min_voxels):
    """Component labels, with the bridge axis chosen to give `expect` parts.

    Each of the three axes is tried in turn and the first that yields exactly
    `expect` components is used. If none does, the isotropic bridge is used and
    the count reported as it falls, so a mask that genuinely does not contain
    `expect` electrodes is not forced to appear as though it does.
    """
    if bridge <= 0:
        lab, _ = label(binary)
        return lab, "none"
    for axis, name in ((0, "x"), (1, "y"), (2, "z")):
        grown, _ = label(_axis_bridge(binary, axis, bridge))
        painted = np.where(binary, grown, 0)
        n = len([v for v in np.unique(painted) if v != 0])
        if n == expect:
            return painted, name
    grown, _ = label(binary_dilation(binary, iterations=bridge))
    return np.where(binary, grown, 0), "isotropic"


def gt_volumes(root, bridge=1, min_voxels=8, expect=4, report=True,
               smooth=False, radius=1, speckle_min=8):
    """Per-electrode volumes from the expert masks, keyed by image tag.

    Gap bridging is applied because mask smoothing left a thin empty layer
    inside some electrodes, so plain connected-component labelling splits one
    electrode into several pieces. The dilation decides GROUPING only; volumes
    are counted on the original voxels, so nothing is added.
    """
    import nibabel as nib
    out = {}
    # The dataset root is sometimes given as the top of a mirrored tree, with
    # the subject folders several levels down. Try the exact layout first and
    # fall back to a recursive search rather than silently finding nothing.
    paths = sorted(glob.glob(os.path.join(root, "sub-*", "unzipped", "*",
                                          "mask.nii.gz")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(root, "**", "sub-*", "unzipped",
                                              "*", "mask.nii.gz"),
                                 recursive=True))
        if paths:
            depth = paths[0][len(root):].count(os.sep) - 4
            print(f"  (subject folders found {depth} level(s) below --gt-root; "
                  f"searched recursively)")
    if not paths:
        paths = sorted(glob.glob(os.path.join(root, "**", "mask.nii.gz"),
                                 recursive=True))
        if paths:
            print(f"  (found {len(paths)} mask.nii.gz by unrestricted "
                  f"recursive search; check these are the expert masks)")
    counts = {}
    axes_used = {}
    smoothing_added, smoothing_before = [0], [0]
    for p in paths:
        tag = os.path.basename(os.path.dirname(p))
        try:
            img = nib.load(p)
            binary = np.asanyarray(img.dataobj) > 0
        except Exception:                                       # noqa: BLE001
            continue
        if smooth:
            n0 = int(binary.sum())
            binary, added = smooth_mask(binary, expect=expect, radius=radius,
                                        bridge=bridge, speckle_min=speckle_min)
            smoothing_added[0] += added
            smoothing_before[0] += n0
        vox = float(abs(np.linalg.det(np.asarray(img.affine)[:3, :3])))
        painted, axis_used = _label_electrodes(binary, expect, bridge,
                                               min_voxels)
        ids = [v for v in np.unique(painted) if v != 0]
        sizes = [int((painted == i).sum()) for i in ids]
        axes_used[axis_used] = axes_used.get(axis_used, 0) + 1
        sizes = sorted((s for s in sizes if s >= min_voxels), reverse=True)
        counts[len(sizes)] = counts.get(len(sizes), 0) + 1
        if len(sizes) != expect:
            continue                       # only clean masks define the reference
        out[tag] = [s * vox for s in sizes]
    if smooth and report and smoothing_before[0]:
        pct = smoothing_added[0] / smoothing_before[0] * 100
        print(f"  smoothing added {smoothing_added[0]:,} voxel(s) to the expert "
              f"masks ({pct:.3f}% of annotated volume)")
        print("  the reference volume below therefore INCLUDES those voxels; "
              "compare against a run without --smooth")
    if report and counts:
        n_ok = counts.get(expect, 0)
        print(f"  bridge axis used: " +
              ", ".join(f"{k} {v}" for k, v in sorted(axes_used.items())))
        print(f"  masks read: {sum(counts.values())}, with exactly {expect} "
              f"electrodes: {n_ok}")
        if n_ok < sum(counts.values()):
            other = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())
                              if k != expect)
            print(f"  other component counts ({other}) are excluded from the "
                  f"reference, since a mask whose electrodes are merged or "
                  f"fragmented would bias the volume it defines")
    return out


def norm_tag(s):
    """sub-XXX_ses-Y_run-ZZ from any of the naming variants in use."""
    m = re.search(r"(sub-\w+?)_(ses-\w+?)_(run-\w+?)(?:_|$)", s + "_")
    return "_".join(m.groups()) if m else s


# ---------------------------------------------------------------------------
# Network tables
# ---------------------------------------------------------------------------

def load_network(tables, net):
    hits = sorted(Path(tables).glob(f"electrode_positions_{net}_*.csv"))
    if not hits:
        return None
    d = pd.read_csv(hits[-1], low_memory=False)
    if not all(c in d.columns for c in VOL_COLS):
        print(f"  {net}: no vol_* columns in {hits[-1].name} — re-run "
              f"01_Extract_coordinate_ablation_final.py to add them")
        return None
    d["tag"] = (d["subject"].astype(str) + "_" + d["session"].astype(str)
                + "_" + d["run"].astype(str))
    return d


def main(a):
    here = Path(__file__).resolve().parent
    tables = Path(a.tables) if a.tables else here / "Tables"
    outdir = Path(a.output_dir) if a.output_dir else here / "Figures"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")

    print(f"\n{'=' * 70}\n  Electrode volume: networks vs expert ground truth\n{'=' * 70}")

    gt = (gt_volumes(a.gt_root, a.bridge, a.min_voxels, expect=a.expect,
                     smooth=a.smooth,
                     radius=a.radius, speckle_min=a.speckle_min)
          if a.gt_root else {})
    if a.gt_root and not gt:
        # Failing here with a clear message beats crashing inside numpy on an
        # empty array three frames deeper.
        raise SystemExit(
            f"\n  No usable ground-truth masks under {a.gt_root}\n"
            f"  Expected: <root>/sub-*/unzipped/<run>/mask.nii.gz\n"
            f"  Check the path -- the subject folders may sit deeper, e.g.\n"
            f"    {a.gt_root}/media/MeMoSLAP_Subjects/derivatives/"
            f"automated_electrode_extraction\n"
            f"  Verify with:\n"
            f"    find {a.gt_root} -name mask.nii.gz | head\n"
            f"  A mask is only used if it contains exactly {a.expect} separate "
            f"electrodes after gap bridging; if masks exist but none qualify, "
            f"raise --bridge or lower --expect.")
    if a.gt_root:
        flat = [v for vs in gt.values() for v in vs]
        gt_med = float(np.median(flat)) if flat else float("nan")
        gt_cv = [float(np.std(v) / np.mean(v)) for v in gt.values()]
        print(f"  Ground truth : {len(gt)} masks with four separate electrodes, "
              f"{len(flat)} electrodes")
        print(f"                 median volume {gt_med:.0f} mm3  "
              f"(IQR {np.percentile(flat,25):.0f}-{np.percentile(flat,75):.0f})")
        print(f"                 within-mask CV median {np.median(gt_cv):.3f}"
              f"   <- the annotation's own spread, a floor for any network")
    else:
        gt_med, gt_cv = float("nan"), []
        print("  Ground truth : not supplied (--gt-root); global comparison only")

    rows, per_net = [], {}
    for net in a.networks:
        d = load_network(tables, net)
        if d is None:
            continue
        per_net[net] = d
        vols = d[VOL_COLS].to_numpy(dtype=float).ravel()
        vols = vols[np.isfinite(vols)]
        cv = (d[VOL_COLS].std(axis=1, ddof=0) /
              d[VOL_COLS].mean(axis=1)).replace([np.inf, -np.inf], np.nan).dropna()
        r = dict(network=net, n_images=len(d), n_electrodes=len(vols),
                 median=float(np.median(vols)),
                 p5=float(np.percentile(vols, 5)),
                 p95=float(np.percentile(vols, 95)),
                 within_cv_median=float(np.median(cv)),
                 within_cv_p95=float(np.percentile(cv, 95)))
        if gt:
            r["bias_vs_gt_mm3"] = r["median"] - gt_med
            r["bias_vs_gt_pct"] = (r["median"] - gt_med) / gt_med * 100
        rows.append(r)

    if not rows:
        raise SystemExit("No network tables with volume columns were found.")
    summary = pd.DataFrame(rows).sort_values("median")

    print(f"\n  ── global comparison (all images) ──")
    cols = ["network", "n_electrodes", "median", "p5", "p95",
            "within_cv_median", "within_cv_p95"]
    if gt:
        cols += ["bias_vs_gt_mm3", "bias_vs_gt_pct"]
    print(summary[cols].to_string(index=False,
                                  float_format=lambda x: f"{x:.3f}"
                                  if abs(x) < 10 else f"{x:.0f}"))

    # ── paired comparison on the annotated images ────────────────────────
    paired_rows = []
    if gt:
        gtn = {norm_tag(k): v for k, v in gt.items()}
        print(f"\n  ── paired comparison (annotated images only) ──")
        for net, d in per_net.items():
            pairs = []
            for _, r in d.iterrows():
                g = gtn.get(norm_tag(r["tag"]))
                if g is None:
                    continue
                nv = sorted([r[c] for c in VOL_COLS], reverse=True)
                if any(pd.isna(x) for x in nv):
                    continue
                pairs += list(zip(nv, sorted(g, reverse=True)))
            if len(pairs) < 8:
                print(f"  {net:<16} too few matched images ({len(pairs)//4})")
                continue
            n_arr = np.array([p[0] for p in pairs])
            g_arr = np.array([p[1] for p in pairs])
            diff = n_arr - g_arr
            try:
                w, pv = stats.wilcoxon(n_arr, g_arr)
            except ValueError:
                pv = float("nan")
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            paired_rows.append(dict(
                network=net, n_images=len(pairs) // 4, n_electrodes=len(pairs),
                median_net=float(np.median(n_arr)),
                median_gt=float(np.median(g_arr)),
                mean_diff=float(diff.mean()),
                ci_lo=float(diff.mean() - 1.96 * se),
                ci_hi=float(diff.mean() + 1.96 * se),
                pct_diff=float(diff.mean() / g_arr.mean() * 100),
                wilcoxon_p=float(pv)))
        if paired_rows:
            pr = pd.DataFrame(paired_rows).sort_values("mean_diff", key=abs)
            print(pr.to_string(index=False,
                               float_format=lambda x: f"{x:.4g}"))
            print("\n  mean_diff is network minus expert, in mm3; a positive "
                  "value means the\n  network segments larger electrodes than "
                  "the annotator drew.")

    # ── figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    order = summary.network.tolist()
    data = [per_net[n][VOL_COLS].to_numpy(dtype=float).ravel() for n in order]
    data = [d[np.isfinite(d)] for d in data]
    parts = axes[0].violinplot(data, positions=range(len(order)), widths=0.8,
                               showextrema=False, showmedians=True)
    for b in parts["bodies"]:
        b.set_facecolor("#cfd8e3"); b.set_alpha(0.75)
    if gt:
        axes[0].axhline(gt_med, color="#c62828", lw=2, ls="--",
                        label=f"expert median {gt_med:.0f} mm$^3$")
        axes[0].legend()
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(order, rotation=20, ha="right")
    axes[0].set_ylabel("Per-electrode volume (mm$^3$)")
    axes[0].grid(axis="y", alpha=0.25)

    cvs = [(per_net[n][VOL_COLS].std(axis=1, ddof=0) /
            per_net[n][VOL_COLS].mean(axis=1)).replace(
                [np.inf, -np.inf], np.nan).dropna().to_numpy() for n in order]
    parts = axes[1].violinplot(cvs, positions=range(len(order)), widths=0.8,
                               showextrema=False, showmedians=True)
    for b in parts["bodies"]:
        b.set_facecolor("#cfd8e3"); b.set_alpha(0.75)
    if gt_cv:
        axes[1].axhline(float(np.median(gt_cv)), color="#c62828", lw=2, ls="--",
                        label=f"expert median CV {np.median(gt_cv):.3f}")
        axes[1].legend()
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(order, rotation=20, ha="right")
    axes[1].set_ylabel("Within-image volume CV")
    axes[1].set_ylim(0, min(1.0, max(np.percentile(c, 99) for c in cvs) * 1.1))
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fp = outdir / f"volume_comparison_{stamp}.png"
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary.to_csv(outdir / f"volume_summary_{stamp}.csv", index=False)
    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(
            outdir / f"volume_paired_{stamp}.csv", index=False)
    print(f"\n  Wrote {fp.name}, volume_summary_{stamp}.csv"
          f"{', volume_paired_' + stamp + '.csv' if paired_rows else ''}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--networks", nargs="+", required=True)
    ap.add_argument("--tables", default=None)
    ap.add_argument("--gt-root", dest="gt_root", default=None,
                    help="dataset root holding sub-*/unzipped/<run>/mask.nii.gz")
    ap.add_argument("--output-dir", dest="output_dir", default=None)
    ap.add_argument("--min-voxels", dest="min_voxels", type=int, default=8,
                    help="ignore connected components smaller than this. The "
                         "default matches inspect_gt_masks.py; at 1, a few "
                         "stray annotation voxels are counted as a fifth "
                         "electrode and the mask is excluded from the "
                         "reference.")
    ap.add_argument("--expect", type=int, default=4,
                    help="electrodes expected per expert mask; masks with a "
                         "different count are excluded from the reference")
    ap.add_argument("--smooth", action="store_true",
                    help="re-smooth the expert masks before measuring: remove "
                         "speckle and close intra-electrode gaps per electrode. "
                         "OFF by default, because closing adds voxels and those "
                         "voxels enter the reference volume that every network "
                         "is compared against; gap bridging already fixes the "
                         "grouping without adding anything. Use it to see how "
                         "much the smoothing would change the reference.")
    ap.add_argument("--radius", type=int, default=1,
                    help="closing radius in voxels, with --smooth")
    ap.add_argument("--speckle-min", dest="speckle_min", type=int, default=8,
                    help="with --smooth, isolated components smaller than this "
                         "are deleted as annotation speckle")
    ap.add_argument("--bridge", type=int, default=1,
                    help="dilation iterations used to group electrode fragments "
                         "in the expert masks; volumes are counted on the "
                         "original voxels either way")
    main(ap.parse_args())
