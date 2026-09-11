#!/usr/bin/env python3
"""
==============================================================
  07_check_affines.py
==============================================================
  Audits the NIfTI geometry metadata to find coordinate-space
  mismatches.

  WHY
  ---
  Extracted electrode coordinates are produced by applying the
  affine of the inference volume to a voxel index, so they live
  in whatever space that affine describes. The SimNIBS baseline
  lives in the space of the head mesh, which was built from a
  T1. If those two spaces differ, every electrode is displaced
  by the same rigid transform -- which is exactly the signature
  seen in some projects: a large residual with the montage shape
  perfectly intact.

  Coregistration.m reslices each PETRA onto the ses-base T1 with
  SPM's coreg:estwrite. Reslicing writes the source onto the
  REFERENCE grid, so a correctly coregistered PETRA must have the
  same shape, voxel size and affine as that T1. Any difference is
  either a coregistration that did not run, one that used a
  different reference, or a file that was never resliced.

  WHAT IT CHECKS
    * shape, voxel size, orientation codes and origin of every file
    * whether each coregistered PETRA matches its ses-base T1
    * whether the inference mask matches its own PETRA
    * whether qform and sform agree within a file, and their codes
    * per-project summary, so a systematic problem is visible

--------------------------------------------------------------
  HOW TO USE
      python 07_check_affines.py --network b_no_attention
      python 07_check_affines.py --network b_no_attention --subjects sub-3001 sub-3002
      python 07_check_affines.py --network b_no_attention --csv affine_audit.csv
==============================================================
"""

import os
import re
import csv
import glob
import argparse
from collections import defaultdict

import numpy as np
import nibabel as nib

DEFAULT_IMAGES = '/media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction'


def project_of(sub):
    m = re.search(r"sub-(\d)", sub)
    return f"P{m.group(1)}" if m else "unknown"


def geom(path):
    """Geometry metadata of one NIfTI, without loading the voxel data."""
    img = nib.load(path)
    hdr = img.header
    aff = img.affine
    return dict(
        path=path, name=os.path.basename(path),
        shape=tuple(int(s) for s in img.shape[:3]),
        zooms=tuple(round(float(z), 4) for z in hdr.get_zooms()[:3]),
        axcodes="".join(nib.aff2axcodes(aff)),
        origin=tuple(round(float(v), 3) for v in aff[:3, 3]),
        affine=aff,
        qform_code=int(hdr['qform_code']), sform_code=int(hdr['sform_code']),
    )


def affine_close(a, b, tol=1e-3):
    return bool(np.allclose(np.asarray(a), np.asarray(b), atol=tol))


def compare(a, b, tol=1e-3):
    """Which aspects of two geometries differ."""
    d = []
    if a['shape'] != b['shape']:
        d.append(f"shape {a['shape']} vs {b['shape']}")
    if a['zooms'] != b['zooms']:
        d.append(f"voxel {a['zooms']} vs {b['zooms']}")
    if a['axcodes'] != b['axcodes']:
        d.append(f"orientation {a['axcodes']} vs {b['axcodes']}")
    if not affine_close(a['affine'], b['affine'], tol):
        off = np.asarray(a['affine'])[:3, 3] - np.asarray(b['affine'])[:3, 3]
        rot = np.asarray(a['affine'])[:3, :3] - np.asarray(b['affine'])[:3, :3]
        d.append(f"affine differs (origin by {np.linalg.norm(off):.2f} mm, "
                 f"direction by {np.abs(rot).max():.4f})")
    return d


def main(a):
    subs = sorted(glob.glob(os.path.join(a.images, "sub-*")))
    if a.subjects:
        keep = set(a.subjects)
        subs = [s for s in subs if os.path.basename(s) in keep]
    if not subs:
        raise SystemExit(f"No subject folders under {a.images}")

    print(f"\n{'=' * 66}\n  NIfTI geometry audit\n{'=' * 66}")
    print(f"  Images  : {a.images}")
    print(f"  Network : {a.network}")
    print(f"  Subjects: {len(subs)}\n{'=' * 66}\n")

    rows = []
    per_proj = defaultdict(lambda: dict(n=0, no_t1=0, petra_mismatch=0,
                                        mask_mismatch=0, qs_mismatch=0,
                                        ok=0, shapes=set(), origins=[]))

    for sd in subs:
        sub = os.path.basename(sd)
        proj = project_of(sub)
        unz = os.path.join(sd, "unzipped")

        t1s = glob.glob(os.path.join(unz, f"{sub}_ses-base_acq-mprage_T1w.nii"))
        t1 = geom(t1s[0]) if t1s else None
        if t1 is None:
            per_proj[proj]['no_t1'] += 1

        petras = sorted(glob.glob(os.path.join(
            unz, "rsub-*_acq-petra_run-*_PDw.nii")))
        for p in petras:
            per_proj[proj]['n'] += 1
            try:
                g = geom(p)
            except Exception as exc:                            # noqa: BLE001
                rows.append(dict(subject=sub, project=proj, file=os.path.basename(p),
                                 issue=f"unreadable: {type(exc).__name__}"))
                continue
            per_proj[proj]['shapes'].add(g['shape'])
            per_proj[proj]['origins'].append(g['origin'])

            issues = []
            if t1 is not None:
                diff = compare(g, t1, a.tol)
                if diff:
                    issues.append("PETRA does not match ses-base T1: " +
                                  "; ".join(diff))
                    per_proj[proj]['petra_mismatch'] += 1
            else:
                issues.append("no ses-base T1 to compare against")

            mask = p.replace("_PDw.nii", f"_PDw_inference_{a.network}.nii.gz")
            if os.path.isfile(mask):
                try:
                    gm = geom(mask)
                    dm = compare(gm, g, a.tol)
                    if dm:
                        issues.append("mask does not match its PETRA: " +
                                      "; ".join(dm))
                        per_proj[proj]['mask_mismatch'] += 1
                except Exception as exc:                        # noqa: BLE001
                    issues.append(f"mask unreadable: {type(exc).__name__}")

            # qform and sform describe the same space in two encodings; if they
            # disagree, which one a reader uses changes the coordinates.
            try:
                img = nib.load(p)
                q = img.get_qform(); s = img.get_sform()
                if g['qform_code'] and g['sform_code'] and not affine_close(q, s, 1e-2):
                    issues.append(f"qform and sform disagree "
                                  f"(origin differs by "
                                  f"{np.linalg.norm(q[:3,3]-s[:3,3]):.2f} mm)")
                    per_proj[proj]['qs_mismatch'] += 1
            except Exception:                                   # noqa: BLE001
                pass

            if not issues:
                per_proj[proj]['ok'] += 1
            rows.append(dict(
                subject=sub, project=proj, file=os.path.basename(p),
                shape="x".join(map(str, g['shape'])),
                zooms="x".join(f"{z:g}" for z in g['zooms']),
                orientation=g['axcodes'],
                origin_x=g['origin'][0], origin_y=g['origin'][1],
                origin_z=g['origin'][2],
                qform_code=g['qform_code'], sform_code=g['sform_code'],
                t1_shape=("x".join(map(str, t1['shape'])) if t1 else ""),
                t1_origin_x=(t1['origin'][0] if t1 else ""),
                matches_t1="yes" if (t1 and not compare(g, t1, a.tol)) else "no",
                issue="; ".join(issues)))

    print(f"{'project':<9}{'files':>7}{'match T1':>10}{'PETRA≠T1':>10}"
          f"{'mask≠PETRA':>12}{'q≠s':>6}{'no T1':>7}   distinct shapes")
    for proj in sorted(per_proj):
        v = per_proj[proj]
        print(f"  {proj:<7}{v['n']:>7}{v['ok']:>10}{v['petra_mismatch']:>10}"
              f"{v['mask_mismatch']:>12}{v['qs_mismatch']:>6}{v['no_t1']:>7}"
              f"   {len(v['shapes'])}")

    print("\n  ── origin spread per project (mm) ──")
    print("  A coregistered set should share the reference grid, so the origin")
    print("  varies between subjects but not within one.")
    for proj in sorted(per_proj):
        o = np.array(per_proj[proj]['origins'], dtype=float)
        if len(o) < 2:
            continue
        print(f"  {proj:<7} origin SD ({o[:,0].std():7.2f},{o[:,1].std():7.2f},"
              f"{o[:,2].std():7.2f})   range X "
              f"[{o[:,0].min():.1f}, {o[:,0].max():.1f}]")

    bad = [r for r in rows if r.get('issue')]
    if bad:
        print(f"\n  {len(bad)} of {len(rows)} file(s) with a geometry issue. "
              f"First few:")
        for r in bad[:12]:
            print(f"    {r['file'][:58]:<58} {r['issue'][:70]}")
        if len(bad) > 12:
            print(f"    ... and {len(bad)-12} more")
    else:
        print("\n  No geometry mismatches found.")

    if a.csv:
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\n  Wrote {a.csv}  ({len(rows)} rows)")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default="b_no_attention")
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="restrict to these subject folders")
    ap.add_argument("--csv", default="affine_audit.csv")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="tolerance for comparing affine matrices")
    main(ap.parse_args())
