#!/usr/bin/env python3
"""
==============================================================
  inspect_gt_masks.py
==============================================================
  Names the expert masks that do not contain exactly four
  separate electrodes, says what is wrong with each, and renders
  them for visual inspection.

  There are two distinct faults and they need different fixes,
  so the script distinguishes them rather than reporting a count:

    TOO FEW components  two or more electrodes are annotated as
                        one connected blob. No amount of
                        smoothing or gap-closing recovers this;
                        the electrodes were drawn touching, or a
                        bridge of voxels joins them. Fixing it
                        means editing the annotation.

    TOO MANY components an electrode is split into pieces, from
                        a thin gap or from stray voxels. This is
                        recoverable automatically.

  For each affected mask the report gives the component volumes.
  A merged pair shows one component at roughly twice the others,
  which is the quickest confirmation of the diagnosis.

--------------------------------------------------------------
  HOW TO USE
      python inspect_gt_masks.py --root <dataset_RU path>
      python inspect_gt_masks.py --root <...> --render --out-dir gt_check
==============================================================
"""

import os
import glob
import argparse

import numpy as np
import nibabel as nib
from scipy.ndimage import label, center_of_mass, binary_dilation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _axis_bridge(binary, axis, iters=1):
    """Dilate along ONE axis only.

    The gap left inside an electrode runs through its thickness; the space
    between two electrodes lies in the plane of the scalp. An isotropic
    dilation large enough to close the gap also reaches sideways and fuses
    neighbouring electrodes, which is why these masks appear to have three
    components under bridging and four without it.
    """
    st = np.zeros((3, 3, 3), dtype=bool)
    sl = [1, 1, 1]
    for k in (0, 1, 2):
        sl[axis] = k
        st[tuple(sl)] = True
    return binary_dilation(binary, st, iterations=iters)


def components(binary, bridge=0, min_voxels=1, expect=None):
    if bridge > 0 and expect:
        # try each axis and take the first that gives the expected count
        for axis in (0, 1, 2):
            grown, _ = label(_axis_bridge(binary, axis, bridge))
            painted = np.where(binary, grown, 0)
            ids = [v for v in np.unique(painted) if v != 0
                   and (painted == v).sum() >= min_voxels]
            if len(ids) == expect:
                return [(int((painted == i).sum()), center_of_mass(painted == i))
                        for i in ids]
    if bridge > 0:
        grown, _ = label(binary_dilation(binary, iterations=bridge))
        painted = np.where(binary, grown, 0)
        ids = [v for v in np.unique(painted) if v != 0]
        return [(int((painted == i).sum()), center_of_mass(painted == i))
                for i in ids if (painted == i).sum() >= min_voxels]
    lab, n = label(binary)
    return [(int((lab == i).sum()), center_of_mass(lab == i))
            for i in range(1, n + 1) if (lab == i).sum() >= min_voxels]


def render(anat_path, mask_path, comps, out_png, title):
    """Three slices through each component, so a merged pair is visible."""
    mimg = nib.load(mask_path)
    mask = np.asanyarray(mimg.dataobj) > 0
    if anat_path and os.path.isfile(anat_path):
        anat = np.asanyarray(nib.load(anat_path).dataobj).astype(np.float32)
        lo, hi = np.percentile(anat, [1, 99.5])
        anat = np.clip((anat - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        anat = np.zeros_like(mask, dtype=np.float32)

    lab, n = label(mask)
    rows = max(len(comps), 1)
    fig, axes = plt.subplots(rows, 3, figsize=(8, 2.7 * rows), squeeze=False)
    for r, (vox, ctr) in enumerate(comps):
        c = [int(round(v)) for v in ctr]
        for k, (name, axis) in enumerate((("Sagittal", 0), ("Coronal", 1),
                                          ("Axial", 2))):
            ax = axes[r][k]
            idx = int(np.clip(c[axis], 0, anat.shape[axis] - 1))
            ax.imshow(np.take(anat, idx, axis=axis).T, cmap="gray",
                      origin="lower", interpolation="nearest")
            m = np.take(mask, idx, axis=axis).T
            if m.any():
                ax.imshow(np.ma.masked_where(~m, m), cmap="autumn", alpha=0.45,
                          origin="lower", interpolation="nearest")
                ax.contour(m, levels=[0.5], colors="#ff2020", linewidths=0.8)
            if r == 0:
                ax.set_title(name, fontsize=9)
            if k == 0:
                ax.text(0.03, 0.95, f"component {r + 1}: {vox} vox",
                        transform=ax.transAxes, fontsize=8, color="#ffe08a",
                        va="top", bbox=dict(boxstyle="round,pad=0.2",
                                            facecolor="#000000cc",
                                            edgecolor="none"))
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def main(a):
    root = os.path.expanduser(a.root)
    masks = sorted(glob.glob(os.path.join(root, "sub-*", "unzipped", "*",
                                          "mask.nii.gz")))
    if not masks:
        masks = sorted(glob.glob(os.path.join(root, "**", "sub-*", "unzipped",
                                              "*", "mask.nii.gz"),
                                 recursive=True))
    if not masks:
        raise SystemExit(f"No mask.nii.gz under {root}")

    print(f"\n{'=' * 70}\n  Ground-truth masks without exactly "
          f"{a.expect} electrodes\n{'=' * 70}")
    print(f"  Root  : {root}\n  Masks : {len(masks)}\n{'=' * 70}\n")

    bad, ok = [], 0
    for p in masks:
        tag = os.path.basename(os.path.dirname(p))
        try:
            img = nib.load(p)
            binary = np.asanyarray(img.dataobj) > 0
        except Exception as exc:                                # noqa: BLE001
            print(f"  {tag:<30} UNREADABLE: {type(exc).__name__}")
            continue
        vox = float(abs(np.linalg.det(np.asarray(img.affine)[:3, :3])))
        raw = components(binary, 0, a.min_voxels)
        grp = components(binary, a.bridge, a.min_voxels, a.expect)
        if len(grp) == a.expect:
            ok += 1
            continue

        kind = "TOO FEW" if len(grp) < a.expect else "TOO MANY"
        vols = sorted((v * vox for v, _ in grp), reverse=True)
        med = float(np.median(vols)) if vols else 0.0
        ratio = (max(vols) / med) if med else float("nan")
        print(f"  {tag:<30} {kind}: {len(grp)} component(s) "
              f"(raw {len(raw)})")
        print(f"      volumes mm3: " +
              ", ".join(f"{v:.0f}" for v in vols))
        if kind == "TOO FEW":
            hint = ("  -> consistent with two electrodes merged into one"
                    if ratio > 1.6 else "  -> not an obvious merge; inspect")
            print(f"      largest / median = {ratio:.2f}{hint}")
            print("      Not recoverable by smoothing: the annotation joins "
                  "two electrodes.")
        else:
            print("      Recoverable: raise --bridge or remove speckle.")
        bad.append((tag, p, kind, grp))

    print(f"\n{'=' * 70}")
    print(f"  with exactly {a.expect} electrodes : {ok} of {len(masks)}")
    print(f"  needing attention            : {len(bad)}")
    print(f"{'=' * 70}\n")

    if a.render and bad:
        out = os.path.expanduser(a.out_dir)
        os.makedirs(out, exist_ok=True)
        for tag, p, kind, grp in bad:
            anat = None
            d = os.path.dirname(os.path.dirname(p))
            sub = tag.split("_")[0]
            for cand in glob.glob(os.path.join(d, f"{sub}_*_PDw.nii")) + \
                        glob.glob(os.path.join(d, f"r{sub}_*_PDw.nii")):
                anat = cand
                break
            png = os.path.join(out, f"{tag}_{kind.replace(' ', '')}.png")
            try:
                render(anat, p, grp, png,
                       f"{tag} — {kind}: {len(grp)} component(s)")
                print(f"  wrote {os.path.basename(png)}")
            except Exception as exc:                            # noqa: BLE001
                print(f"  could not render {tag}: {type(exc).__name__}: {exc}")
        print(f"\n  Images in {out}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--expect", type=int, default=4)
    ap.add_argument("--bridge", type=int, default=1,
                    help="dilation used to group fragments of one electrode; "
                         "grouping only, nothing is added")
    ap.add_argument("--min-voxels", dest="min_voxels", type=int, default=8,
                    help="ignore components smaller than this")
    ap.add_argument("--render", action="store_true",
                    help="write a PNG per affected mask")
    ap.add_argument("--out-dir", dest="out_dir", default="gt_check")
    main(ap.parse_args())
