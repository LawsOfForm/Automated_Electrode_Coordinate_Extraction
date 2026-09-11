#!/usr/bin/env python3
"""
compare_loss.py — paired per-fold comparison of two loss functions.

Reads the per-case CSVs in ablation_results/ and pairs each new-loss fold
against the dicefocal fold of the SAME index. Because folds are seed-locked
they contain exactly the same test participants, so this is a paired
comparison, not two independent means.

The headline number is NOT mean Dice. It is the count of ZERO-DICE CASES --
complete prediction collapses. Those are what produce inf Hausdorff, what
made 7 of 24 folds unusable, and what a reviewer will notice. If a loss
removes them, that loss is the fix.

Usage:
    python3 compare_loss.py
    python3 compare_loss.py --network proposed --loss dicece
    python3 compare_loss.py --results-dir ~/ablation/ablation_results
"""

from __future__ import annotations

import argparse
import csv
import glob
import os.path as op
import re
import statistics as st


def load_cases(path):
    """Return list of dicts with float dice/hausdorff/iou (inf preserved)."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            def num(k):
                v = (r.get(k) or "").strip()
                if v.lower() in ("inf", "+inf"):
                    return float("inf")
                if v.lower() in ("nan", ""):
                    return float("nan")
                try:
                    return float(v)
                except ValueError:
                    return float("nan")
            rows.append({"file": r.get("file", ""), "subject": r.get("subject", ""),
                         "dice": num("dice"), "hausdorff": num("hausdorff"),
                         "iou": num("iou")})
    return rows


def find_folds(results_dir, network, suffix):
    """Map fold index -> per-case CSV path for a given tag suffix."""
    out = {}
    pat = op.join(results_dir, f"*_cv5f*_seed*_{suffix}_test_per_case.csv")
    for p in glob.glob(pat):
        m = re.search(r"_cv5f(\d+)_seed", op.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def summarize(rows):
    d = [r["dice"] for r in rows]
    zeros = [r for r in rows if r["dice"] == 0.0]
    finite_hd = [r["hausdorff"] for r in rows
                 if r["hausdorff"] == r["hausdorff"] and r["hausdorff"] != float("inf")]
    return {
        "n": len(rows),
        "dice": st.mean(d) if d else float("nan"),
        "zeros": len(zeros),
        "zero_subjects": sorted({r["subject"] for r in zeros}),
        "n_inf_hd": sum(1 for r in rows if r["hausdorff"] == float("inf")),
        "hd_finite": st.mean(finite_hd) if finite_hd else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",
                    default=op.expanduser("~/ablation/ablation_results"))
    ap.add_argument("--network", default="proposed")
    ap.add_argument("--loss", default="dicece", help="the NEW loss being tested")
    args = ap.parse_args()

    base = find_folds(args.results_dir, args.network, args.network)
    new = find_folds(args.results_dir, args.network, f"{args.network}_{args.loss}")

    if not new:
        print(f"No {args.loss} runs found for '{args.network}' in {args.results_dir}.")
        print(f"Expected files like: *_cv5f<F>_seed*_{args.network}_{args.loss}"
              f"_test_per_case.csv")
        return
    if not base:
        print(f"No baseline (dicefocal) runs found for '{args.network}'.")
        return

    shared = sorted(set(base) & set(new))
    if not shared:
        print(f"No overlapping folds. baseline folds={sorted(base)}, "
              f"{args.loss} folds={sorted(new)}")
        return

    print(f"Paired comparison: {args.network}   dicefocal  vs  {args.loss}")
    print(f"folds compared: {shared}\n")
    print(f"{'fold':<6}{'dice focal':>12}{'dice ' + args.loss:>14}{'delta':>9}"
          f"{'zeros focal':>13}{'zeros new':>11}")

    db, dn, zb, zn = [], [], 0, 0
    for f in shared:
        b = summarize(load_cases(base[f]))
        n = summarize(load_cases(new[f]))
        db.append(b["dice"]); dn.append(n["dice"])
        zb += b["zeros"]; zn += n["zeros"]
        print(f"{f:<6}{b['dice']:>12.4f}{n['dice']:>14.4f}"
              f"{n['dice'] - b['dice']:>+9.4f}{b['zeros']:>13}{n['zeros']:>11}")
        if b["zero_subjects"] or n["zero_subjects"]:
            print(f"        collapsed subjects -- focal: {b['zero_subjects'] or 'none'}"
                  f" | {args.loss}: {n['zero_subjects'] or 'none'}")

    print(f"\nmean Dice : focal {st.mean(db):.4f}  ->  {args.loss} {st.mean(dn):.4f}"
          f"  ({st.mean(dn) - st.mean(db):+.4f})")
    print(f"ZERO-DICE CASES : focal {zb}  ->  {args.loss} {zn}")

    print()
    if zn == 0 and zb > 0:
        print("VERDICT: the collapses are gone. The loss was the problem.")
        print("  -> re-run the FULL ablation (all 5 variants x 5 folds) under "
              f"--loss {args.loss} before putting any table in the paper.")
    elif zn < zb:
        print(f"VERDICT: fewer collapses ({zb} -> {zn}), but not eliminated.")
        print("  -> the loss is part of it. Consider also raising the foreground "
              "weight, or check whether the remaining cases share a property.")
    elif zn >= zb:
        print("VERDICT: no improvement in collapse count.")
        print("  -> the loss is likely NOT the cause. Next suspects: the "
              "learning rate, or genuinely hard cases. In that case the current "
              "ablation numbers stand and should be reported as they are.")


if __name__ == "__main__":
    main()
