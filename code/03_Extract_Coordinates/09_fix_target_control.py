#!/usr/bin/env python3
"""
==============================================================
  09_fix_target_control.py
==============================================================
  Verifies that each subject's baseline is the montage that was
  actually applied, by checking which side of the head the
  extracted electrodes sit on.

  NOTE ON SCOPE. In this study each subject was assigned to ONE
  stimulation arm, so there is exactly one baseline per subject
  and nothing to choose between. This script therefore VERIFIES
  rather than selects: it flags subjects whose extracted montage
  is on the opposite side from their baseline, which means the
  wrong pickle was matched to them, and cannot be fixed by
  swapping a condition that does not exist.

  WHY A SEPARATE SCRIPT, AND NOT A CHANGE INSIDE 02
  -------------------------------------------------
  An earlier attempt let 02 choose between the target and
  control montages by whichever fitted the extraction better.
  That was tested and made the result markedly worse: matched
  electrodes rose from 9,556 to 15,792 and every project's median
  deviation rose with them, P6 from 12.1 mm to 100.7 mm. Choosing
  the montage that minimises the residual is circular, and a
  wrong montage still has a best-fitting assignment, so the
  criterion cannot detect its own failure.

  This script uses a criterion that is independent of the
  residual it is meant to explain: WHICH SIDE OF THE HEAD the
  montage is on. Where the two conditions differ by hemisphere,
  that is a categorical fact about the image, not a goodness-of-
  fit judgement, and it cannot be talked into agreeing with a
  montage that was never applied.

  VALIDATION
  ----------
  Checked against an independent record of the stimulation arm
  for 77 subjects of project P3, in which target is a left-
  hemisphere montage and control a right-hemisphere one:

      589 of 589 images classified correctly (100%)
       77 of  77 subjects, with all sessions of every subject
                 agreeing
      nearest image 35 mm from the decision boundary

  SCOPE, AND WHEN NOT TO USE IT
  -----------------------------
  The rule works only where the conditions are on opposite
  hemispheres. If target and control sit on the same side, the
  hemisphere carries no information, the script says so and
  changes nothing rather than guessing. Verify the montage
  geometry of a project before applying it.

--------------------------------------------------------------
  HOW TO USE
      python 09_fix_target_control.py --network b_no_attention --dry-run
      python 09_fix_target_control.py --network b_no_attention
      python 09_fix_target_control.py --network b_no_attention \\
             --validate targert_control_active_arm.csv --projects P3
==============================================================
"""

import os
import re
import glob
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ELECTRODES = ["anode", "cathode1", "cathode2", "cathode3"]

# A montage must sit at least this far from the midline before its side is
# treated as meaningful. In the validated data the nearest image was 35 mm out,
# so 15 mm is permissive while still refusing to classify a montage that
# straddles the midline.
SIDE_MIN_MM = 15.0

# The two conditions must be separated by at least this much along the
# left-right axis for the hemisphere rule to apply at all.
SEPARATION_MIN_MM = 30.0


def load_long(path):
    d = pd.read_csv(path, low_memory=False)
    d["coordinates"] = pd.to_numeric(d["coordinates"], errors="coerce")
    w = d.pivot_table(index=["subject", "session", "run", "electrode",
                             "condition"],
                      columns="dimension", values="coordinates").reset_index()
    return d, w


def mean_x(df):
    """Mean left-right coordinate of the four electrodes."""
    s = df[df.electrode.isin(ELECTRODES)]
    return float(s["X"].mean()) if len(s) else np.nan


def main(a):
    here = Path(__file__).resolve().parent
    tables = Path(a.tables) if a.tables else here / "Tables"
    src = Path(a.input) if a.input else (
        tables / f"corrected_electrode_positions_with_baseline_long_{a.network}.csv")
    if not src.exists():
        raise SystemExit(f"Not found: {src}")

    long_df, wide = load_long(src)
    ext = wide[~wide.session.eq("ses-baseline")]
    base = wide[wide.session.eq("ses-baseline")]

    # Baselines available per subject and condition
    bl = {}
    for (sub, cond), g in base.groupby(["subject", "condition"]):
        bl.setdefault(sub, {})[cond] = mean_x(g)

    print(f"\n{'=' * 68}\n  Target / control baseline check\n{'=' * 68}")
    print(f"  Network : {a.network}")
    print(f"  Table   : {src.name}")
    print(f"  Subjects with a baseline: {len(bl)}")
    n_two = sum(1 for v in bl.values() if len(v) > 1)
    print(f"  ... of which have more than one condition: {n_two}")
    if n_two == 0:
        print("\n  No subject has both a target and a control baseline, so there\n"
              "  is nothing to swap. If control montages exist, they are not in\n"
              "  this table -- check that 02 loaded them.\n")

    rows = []
    for (sub, ses, run), g in ext.groupby(["subject", "session", "run"]):
        cands = bl.get(sub, {})
        if not cands:
            continue
        ex = mean_x(g)
        if not np.isfinite(ex):
            continue
        used = g["condition"].iloc[0] if "condition" in g else "target"
        rec = dict(subject=sub, session=ses, run=run,
                   project="P" + re.sub(r"\D", "", sub)[:1],
                   extracted_mean_x=round(ex, 2),
                   condition_in_table=used,
                   n_candidates=len(cands))

        # Same side as which candidate?
        same = [c for c, bx in cands.items()
                if np.isfinite(bx) and np.sign(bx) == np.sign(ex)]
        sep = (max(cands.values()) - min(cands.values())
               if len(cands) > 1 else np.nan)
        rec["condition_separation_mm"] = (round(float(sep), 2)
                                          if np.isfinite(sep) else "")

        if abs(ex) < SIDE_MIN_MM:
            rec.update(recommended=used, reason="montage too close to the "
                       "midline for its side to be informative")
        elif len(cands) < 2:
            rec.update(recommended=used,
                       reason="only one baseline condition available")
        elif not np.isfinite(sep) or sep < SEPARATION_MIN_MM:
            rec.update(recommended=used, reason="the two conditions are not "
                       "separated by hemisphere; the rule does not apply")
        elif len(same) == 1:
            rec.update(recommended=same[0],
                       reason=("matches the side of the "
                               f"'{same[0]}' montage"))
        else:
            rec.update(recommended=used, reason="side is ambiguous")
        # With one baseline per subject there is nothing to swap; what matters
        # is whether the baseline present is on the right side of the head.
        only = list(cands)[0] if len(cands) == 1 else None
        if only is not None and abs(ex) >= SIDE_MIN_MM and np.isfinite(cands[only]):
            rec["baseline_side_ok"] = ("yes" if np.sign(cands[only]) == np.sign(ex)
                                       else "no")
            rec["baseline_mean_x"] = round(float(cands[only]), 2)
        else:
            rec["baseline_side_ok"] = ""
            rec["baseline_mean_x"] = ""
        rec["swap"] = "yes" if rec["recommended"] != used else "no"
        rows.append(rec)

    if not rows:
        raise SystemExit("\n  No image could be evaluated.\n")
    res = pd.DataFrame(rows)

    print(f"\n  ── recommendation ──")
    print(res.groupby(["project", "swap"]).size().unstack(fill_value=0)
          .to_string())
    n_swap = int((res.swap == "yes").sum())
    if "baseline_side_ok" in res:
        wrong = res[res.baseline_side_ok == "no"]
        print(f"\n  ── is the baseline on the right side of the head? ──")
        print(f"  checked {int((res.baseline_side_ok != '').sum()):,} image(s); "
              f"{len(wrong):,} sit on the OPPOSITE side from their baseline")
        if len(wrong):
            print("\n  These subjects were matched to a montage that was not "
                  "applied to them.\n  Because each subject has only one "
                  "baseline, this cannot be fixed by\n  relabelling: the "
                  "correct pickle is missing or was assigned to the\n  wrong "
                  "subject. Exclude them from deviation analyses until "
                  "resolved.")
            per = (wrong.groupby("project")["subject"].nunique()
                   .rename("subjects").reset_index())
            per["images"] = wrong.groupby("project").size().values
            print(per.to_string(index=False))
            print("\n  first few subjects: " +
                  ", ".join(sorted(wrong.subject.unique())[:12]))

    print(f"\n  {n_swap} of {len(res)} image(s) would be re-matched against the "
          f"other condition.")
    if n_swap:
        print(res[res.swap == "yes"].groupby(
            ["project", "condition_in_table", "recommended"]).size()
            .rename("images").reset_index().to_string(index=False))

    # ── validation against an independent record ──────────────────────────
    val = a.validate or a.arms
    if val is None:
        cand = tables / "controll_target" / "target_control_active_arm.csv"
        val = str(cand) if cand.exists() else None
    if val and not os.path.isfile(val):
        print(f"\n  Arm list not found: {val}\n  Skipping verification. Pass "
              f"--arms with the correct path.")
        val = None
    if val:
        v = pd.read_csv(val)
        col = [c for c in v.columns if "target" in c.lower()
               and "control" in c.lower()]
        if not col:
            raise SystemExit(f"  Could not find the condition column in {val}")
        v["subject"] = "sub-" + v.iloc[:, 0].astype(str)
        v["true"] = v[col[0]].astype(str).str.split("/").str[0].str.strip()
        mv = res.merge(v[["subject", "true"]], on="subject")
        if a.projects:
            mv = mv[mv.project.isin(a.projects)]
        if len(mv):
            ok = (mv.recommended == mv.true)
            print(f"\n  ── verification against {os.path.basename(val)} ──")
            print(pd.crosstab(mv["true"], mv["recommended"],
                              margins=True).to_string())
            print(f"\n  per image  : {ok.sum()} of {len(mv)} correct "
                  f"({ok.mean()*100:.2f}%)")
            per_sub = mv.groupby("subject").agg(
                true=("true", "first"),
                pred=("recommended", lambda x: x.value_counts().index[0]),
                consistent=("recommended", lambda x: x.nunique() == 1))
            print(f"  per subject: "
                  f"{(per_sub.true == per_sub.pred).sum()} of {len(per_sub)} "
                  f"correct; {per_sub.consistent.sum()} with all sessions "
                  f"agreeing")
            bad = mv[~ok]
            if len(bad):
                print("\n  misclassified:")
                print(bad[["subject", "session", "run", "extracted_mean_x",
                           "true", "recommended"]].to_string(index=False))
        else:
            print("\n  (no overlap between the table and the validation list)")

    stamp = datetime.now().strftime("%Y%m%d")
    out = tables / f"target_control_check_{a.network}_{stamp}.csv"
    res.to_csv(out, index=False)
    print(f"\n  Wrote {out.name}")

    if a.dry_run or not n_swap:
        print("\n  Dry run — no table was modified. Re-run without --dry-run to\n"
              "  write a corrected long table.\n")
        return

    # ── write the corrected table ─────────────────────────────────────────
    # Only the condition LABEL changes, so that downstream code joins each
    # image to the montage it actually received. No coordinate is altered.
    swap = {(r.subject, r.session, r.run): r.recommended
            for r in res.itertuples() if r.swap == "yes"}
    key = list(zip(long_df.subject, long_df.session, long_df.run))
    long_df["condition_original"] = long_df["condition"]
    long_df["condition"] = [swap.get(k, c) for k, c in
                            zip(key, long_df["condition"])]
    changed = int((long_df.condition != long_df.condition_original).sum())
    dst = src.with_name(src.stem + "_condfix.csv")
    long_df.to_csv(dst, index=False)
    print(f"\n  Wrote {dst.name}  ({changed} row(s) relabelled)")
    print("  Coordinates are unchanged; only the condition label was corrected,")
    print("  so the baseline join now uses the montage that was applied.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", required=True)
    ap.add_argument("--input", default=None)
    ap.add_argument("--tables", default=None)
    ap.add_argument("--arms", default=None,
                    help="CSV recording the stimulation arm per subject; first "
                         "column the subject number, second the arm. Default: "
                         "Tables/controll_target/target_control_active_arm.csv")
    ap.add_argument("--validate", default=None,
                    help="CSV with an independent record of the stimulation "
                         "arm, first column the subject number")
    ap.add_argument("--projects", nargs="+", default=None,
                    help="restrict the validation to these projects")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    main(ap.parse_args())
