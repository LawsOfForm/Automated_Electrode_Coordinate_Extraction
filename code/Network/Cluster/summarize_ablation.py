#!/usr/bin/env python3
"""
summarize_ablation.py — collapse a whole ablation directory into ONE file.

Scans   <root>/ablation_results/*_report.md
        <root>/ablation_results/*_test_per_case.csv   (preferred, has filenames)
        <root>/logs/*.out                             (run status)

Writes  <root>/ablation_summary_<YYYYMMDD>.md
        Sections: coverage, aggregates, paired loss comparison, per-run detail,
        configuration check, job status, per-case metrics.

Standard library only. No pandas, no numpy.

USAGE
    python3 summarize_ablation.py                  # root = this script's folder
    python3 summarize_ablation.py --root ~/ablation
    python3 summarize_ablation.py --no-per-case    # smaller file
    python3 summarize_ablation.py --csv            # also emit a flat per-fold CSV
"""

import os
import re
import csv
import glob
import argparse
import statistics as st
from datetime import datetime
from collections import defaultdict

INF = float("inf")

# Order matters: 'b_no_attention' is tested before 'proposed', because the
# no-attention tag is  b_proposed_no_attention_..._b_no_attention  which
# contains the substring 'proposed'. Most specific names first.
NETWORKS = ["d_increased", "c_increased", "c_reduced", "b_no_attention",
            "a_baseline", "segresnet", "proposed", "c_normal"]
LOSSES = ["dicefocal", "dicece", "diceloss"]
DEFAULT_LOSS = "dicefocal"      # what ablation_utils uses when --loss is absent


def detect(name, options, default=None):
    for opt in options:
        if opt in name:
            return opt
    return default


def parse_tag(tag):
    """Split an experiment tag into (network, dataset, fold, seed, loss).

    Never returns None for network or loss: an unrecognised tag is labelled
    'UNKNOWN' so it stays visible in the summary instead of crashing the sort.
    """
    net = detect(tag, NETWORKS) or "UNKNOWN"
    loss = detect(tag, LOSSES, default=DEFAULT_LOSS)
    ds = "RU" if "_RU_" in tag else ("HGW" if "_HGW_" in tag else "?")

    m = re.search(r"cv5f(\d)", tag)
    if m:
        fold = int(m.group(1))
    elif "_FINAL_" in tag:
        fold = "FINAL"
    else:
        fold = "single"

    m = re.search(r"seed(\d+)", tag)
    seed = m.group(1) if m else "?"
    return net, ds, fold, seed, loss


def sort_key(k):
    """Type-safe sort key: tuples mix str and int (fold 0..4, 'FINAL', 'single')."""
    return tuple(str(x) for x in k)


def fold_key(f):
    """Numeric folds first and in order; FINAL/single last."""
    return (0, f) if isinstance(f, int) else (1, str(f))


def cell(text, key):
    """Pull the value column out of a 2-column markdown row labelled `key`."""
    m = re.search(r"^\|\s*`?" + re.escape(key) + r"`?\s*\|\s*(.+?)\s*\|\s*$",
                  text, re.M)
    return m.group(1).strip() if m else None


def to_float(x):
    """'0.6640' -> 0.664 ; 'inf' -> inf ; ''/'--'/'nan'/None -> None."""
    if x is None:
        return None
    s = str(x).strip()
    if s in ("", "--", "nan", "NaN"):
        return None
    if s == "inf":
        return INF
    try:
        return float(s)
    except ValueError:
        return None


def parse_report(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    tag = os.path.basename(path).replace("_report.md", "")
    net, ds, fold, seed, loss = parse_tag(tag)

    r = dict(tag=tag, network=net, dataset=ds, fold=fold, seed=seed, loss=loss,
             path=path)

    m = re.search(r"\*Generated:\*\s*(.+)", t)
    r["generated"] = m.group(1).strip() if m else None
    m = re.search(r"\*Host:\*\s*(.+)", t)
    r["host"] = m.group(1).strip() if m else None

    for k, label in [("params", "**Total parameters**"), ("channels", "channels"),
                     ("architecture", "architecture"), ("loss_label", "loss"),
                     ("loss_name", "loss_name"), ("lr", "learning_rate"),
                     ("optimizer", "optimizer"), ("max_iter", "max_iterations"),
                     ("eval_num", "eval_num"), ("is_final", "is_final_model")]:
        r[k] = cell(t, label)

    m = re.search(r"\| Best Dice \(val\) \| ([\d.]+) \|", t)
    r["best_val"] = to_float(m.group(1)) if m else None
    m = re.search(r"\| Iteration of best checkpoint \| (\d+) \|", t)
    r["best_iter"] = int(m.group(1)) if m else None

    # Held-out test block. --final runs have no test set; those stay None.
    blk = (t.split("## Held-out test set results")[-1]
           if "## Held-out test set results" in t else "")
    for k, label in [("dice", "Dice"), ("hd", "Hausdorff"), ("iou", "IoU")]:
        m = re.search(r"^\| " + label + r" \| ([\d.]+|inf|nan) \| ([\d.]+|inf|nan) \|",
                      blk, re.M)
        r[k] = to_float(m.group(1)) if m else None
        r[k + "_sd"] = to_float(m.group(2)) if m else None

    # Per-sample table from the report (fallback when no CSV exists).
    cases = []
    for a, b, c, d in re.findall(
            r"^\| (\d+) \| ([\d.]+) \| ([\d.]+|inf|nan) \| ([\d.]+) \|$", blk, re.M):
        hd = to_float(c)
        cases.append(dict(idx=int(a), dice=float(b),
                          hd=INF if hd is None else hd,
                          iou=float(d), file=None, subject=None))
    r["cases"] = cases
    r["cases_source"] = "report" if cases else "none"
    return r


def attach_cases_from_csv(rec, results_dir):
    """Prefer the per-case CSV: it carries filenames and subject IDs."""
    p = os.path.join(results_dir, rec["tag"] + "_test_per_case.csv")
    if not os.path.isfile(p):
        return
    rows = []
    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            for i, x in enumerate(csv.DictReader(fh)):
                d = to_float(x.get("dice"))
                if d is None:
                    continue
                hd = to_float(x.get("hausdorff"))
                rows.append(dict(idx=i, dice=d, hd=INF if hd is None else hd,
                                 iou=to_float(x.get("iou")) or 0.0,
                                 file=x.get("file"), subject=x.get("subject")))
    except (KeyError, ValueError, OSError):
        return                      # unexpected columns -> keep the report table
    if rows:
        rec["cases"] = rows
        rec["cases_source"] = "csv"


def scan_logs(logs_dir):
    """Map log filename -> (status, approx iterations reached)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(logs_dir, "*.out"))):
        try:
            txt = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if "CANCELLED" in txt and "TIME LIMIT" in txt:
            status = "KILLED (time limit)"
        elif "CANCELLED" in txt:
            status = "CANCELLED"
        elif "[sanity-check] ABORT" in txt:
            status = "SANITY ABORT"
        elif "Traceback" in txt:
            status = "ERROR"
        elif re.search(r"^End: ", txt, re.M) or "Report written" in txt:
            status = "completed"
        else:
            status = "incomplete / running"
        iters = len(re.findall(r"Validation - Dice", txt)) * 500
        out[os.path.basename(p)] = (status, iters)
    return out


def f4(v, nd=4):
    if v is None:
        return "--"
    if v == INF:
        return "inf"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def main(args):
    root = os.path.abspath(os.path.expanduser(args.root))
    results_dir = os.path.join(root, "ablation_results")
    logs_dir = os.path.join(root, "logs")

    if not os.path.isdir(results_dir):
        raise SystemExit(f"No ablation_results/ directory under {root}")

    reports = sorted(glob.glob(os.path.join(results_dir, "*_report.md")))
    if not reports:
        raise SystemExit(f"No *_report.md found in {results_dir}")

    recs, failed = [], []
    for p in reports:
        try:
            r = parse_report(p)
            attach_cases_from_csv(r, results_dir)
            recs.append(r)
        except Exception as exc:                        # noqa: BLE001
            failed.append((os.path.basename(p), f"{type(exc).__name__}: {exc}"))

    logs = scan_logs(logs_dir) if os.path.isdir(logs_dir) else {}

    out_path = args.out or os.path.join(
        root, f"ablation_summary_{datetime.now():%Y%m%d}.md")

    L = []
    A = L.append

    A(f"# Ablation summary — generated {datetime.now():%Y-%m-%d %H:%M}")
    A(f"\nRoot: `{root}`  ")
    A(f"Reports parsed: {len(recs)}  ")
    A(f"Log files scanned: {len(logs)}  ")
    A(f"Per-case source: {sum(1 for r in recs if r['cases_source'] == 'csv')} csv / "
      f"{sum(1 for r in recs if r['cases_source'] == 'report')} report\n")

    if failed:
        A("**Reports that could not be parsed:**\n")
        for name, err in failed:
            A(f"- `{name}` — {err}")
        A("")

    unknown = sorted({r["tag"] for r in recs if r["network"] == "UNKNOWN"})
    if unknown:
        A("**Tags with an unrecognised network name** "
          "(add them to NETWORKS at the top of the script):\n")
        for u in unknown:
            A(f"- `{u}`")
        A("")

    # ---- 1. coverage --------------------------------------------------------
    A("## 1. Coverage (held-out test Dice per fold)\n")
    groups = defaultdict(dict)
    for r in recs:
        groups[(r["network"], r["dataset"], r["loss"])][r["fold"]] = r

    A("| network | dataset | loss | f0 | f1 | f2 | f3 | f4 | non-CV | n |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(groups, key=sort_key):
        net, ds, loss = key
        g = groups[key]
        cells = [f4(g[f]["dice"]) if f in g else "**missing**" for f in range(5)]
        extra = ", ".join(f"{k}:{f4(g[k]['dice'])}"
                          for k in sorted(g, key=fold_key)
                          if not isinstance(k, int))
        n = sum(1 for f in range(5) if f in g)
        A(f"| {net} | {ds} | {loss} | " + " | ".join(cells) +
          f" | {extra or '--'} | {n}/5 |")

    # ---- 2. aggregates ------------------------------------------------------
    A("\n## 2. Aggregates (CV folds only; mean of fold means)\n")
    A("_HD statistics exclude `inf` cases (empty predictions); their count is "
      "the last column._\n")
    A("| network | dataset | loss | n | mean Dice | SD | median case Dice | "
      "median HD | mean IoU | cases<0.3 | inf HD |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(groups, key=sort_key):
        net, ds, loss = key
        folds = [g for f, g in groups[key].items()
                 if isinstance(f, int) and g["dice"] is not None]
        if not folds:
            continue
        fm = [g["dice"] for g in folds]
        allc = [c for g in folds for c in g["cases"]]
        d = [c["dice"] for c in allc]
        h = [c["hd"] for c in allc if c["hd"] != INF]
        iou = [g["iou"] for g in folds if g["iou"] is not None]
        A(f"| {net} | {ds} | {loss} | {len(fm)} | {st.mean(fm):.4f} | "
          f"{(st.stdev(fm) if len(fm) > 1 else 0):.4f} | "
          f"{(st.median(d) if d else 0):.4f} | {(st.median(h) if h else 0):.2f} | "
          f"{(st.mean(iou) if iou else 0):.4f} | "
          f"{sum(1 for x in d if x < 0.3)}/{len(d)} | "
          f"{sum(1 for c in allc if c['hd'] == INF)} |")

    # ---- 2b. paired loss comparison ----------------------------------------
    pairs = []
    for (net, ds, loss), g in groups.items():
        if loss == DEFAULT_LOSS:
            continue
        base = groups.get((net, ds, DEFAULT_LOSS))
        if not base:
            continue
        common = [f for f in range(5)
                  if f in g and f in base
                  and g[f]["dice"] is not None and base[f]["dice"] is not None]
        if len(common) >= 2:
            pairs.append((net, ds, loss, common,
                          [base[f]["dice"] for f in common],
                          [g[f]["dice"] for f in common]))
    if pairs:
        A(f"\n## 2b. Paired loss comparison (identical folds, vs {DEFAULT_LOSS})\n")
        A(f"| network | dataset | alt loss | folds | {DEFAULT_LOSS} | alt | "
          "mean diff |")
        A("|---|---|---|---|---|---|---|")
        for net, ds, loss, common, a, b in sorted(pairs, key=sort_key):
            diff = st.mean([y - x for x, y in zip(a, b)])
            A(f"| {net} | {ds} | {loss} | {len(common)} | {st.mean(a):.4f} | "
              f"{st.mean(b):.4f} | {diff:+.4f} |")
        A("\n_Paired means only. With n=5 folds, run a paired t-test or Wilcoxon "
          "before claiming a difference._")

    # ---- 3. per-run detail --------------------------------------------------
    A("\n## 3. Per-run detail\n")
    A("| tag | net | ds | fold | loss | params | best val | @iter | "
      "test Dice | sd | test HD | test IoU | host |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(recs, key=lambda x: sort_key(
            (x["network"], x["loss"], fold_key(x["fold"])))):
        A(f"| `{r['tag']}` | {r['network']} | {r['dataset']} | {r['fold']} | "
          f"{r['loss']} | {r['params'] or '--'} | {f4(r['best_val'])} | "
          f"{r['best_iter'] or '--'} | {f4(r['dice'])} | {f4(r['dice_sd'])} | "
          f"{f4(r['hd'], 2)} | {f4(r['iou'])} | {r['host'] or '--'} |")

    # ---- 4. configuration consistency --------------------------------------
    A("\n## 4. Configuration check\n")
    A("_Any field with more than one distinct value across runs of the SAME "
      "network is a red flag._\n")
    for key in ["params", "channels", "architecture", "loss_label", "lr",
                "optimizer", "max_iter", "eval_num"]:
        vals = defaultdict(set)
        for r in recs:
            vals[r.get(key)].add(r["network"])
        A(f"\n**{key}**\n")
        for v, nets in sorted(vals.items(), key=lambda x: (-len(x[1]), str(x[0]))):
            A(f"- `{v}` — {', '.join(sorted(nets))}")

    A("\n**Parameter count per network** (should be exactly one value each)\n")
    pc = defaultdict(set)
    for r in recs:
        if r["params"]:
            pc[r["network"]].add(r["params"])
    for net in sorted(pc):
        vals = sorted(pc[net])
        flag = "" if len(vals) == 1 else "  <-- INCONSISTENT"
        A(f"- {net}: {', '.join(vals)}{flag}")

    # ---- 5. job status ------------------------------------------------------
    if logs:
        A("\n## 5. Job status from logs\n")
        bad = {k: v for k, v in logs.items() if v[0] != "completed"}
        A(f"Completed: {len(logs) - len(bad)} / {len(logs)}\n")
        if bad:
            A("| log | status | approx iterations |")
            A("|---|---|---|")
            for k in sorted(bad):
                A(f"| `{k}` | {bad[k][0]} | {bad[k][1]} |")
        else:
            A("All scanned logs completed.")

    # ---- 6. per-case --------------------------------------------------------
    if not args.no_per_case:
        A("\n## 6. Per-case test metrics\n")
        A("| network | loss | fold | idx | subject | file | Dice | HD | IoU |")
        A("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(recs, key=lambda x: sort_key(
                (x["network"], x["loss"], fold_key(x["fold"])))):
            for c in r["cases"]:
                A(f"| {r['network']} | {r['loss']} | {r['fold']} | {c['idx']} | "
                  f"{c['subject'] or '--'} | {c['file'] or '--'} | "
                  f"{c['dice']:.4f} | {f4(c['hd'], 2)} | {c['iou']:.4f} |")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    # ---- optional flat CSV --------------------------------------------------
    if args.csv:
        csv_path = os.path.splitext(out_path)[0] + "_folds.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["tag", "network", "dataset", "loss", "fold", "seed",
                        "params", "best_val", "best_iter", "test_dice",
                        "test_dice_sd", "test_hd", "test_iou", "host"])
            for r in sorted(recs, key=lambda x: sort_key(
                    (x["network"], x["loss"], fold_key(x["fold"])))):
                w.writerow([r["tag"], r["network"], r["dataset"], r["loss"],
                            r["fold"], r["seed"], r["params"], r["best_val"],
                            r["best_iter"], r["dice"], r["dice_sd"], r["hd"],
                            r["iou"], r["host"]])
        print(f"Wrote {csv_path}")

    print(f"\nWrote {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)")
    print(f"  {len(recs)} reports, {sum(len(r['cases']) for r in recs)} test cases")
    if failed:
        print(f"  {len(failed)} report(s) failed to parse — listed in the summary")
    if unknown:
        print(f"  {len(unknown)} tag(s) with unrecognised network name")
    missing = [(k, f) for k, g in groups.items() for f in range(5) if f not in g]
    if missing:
        print(f"  {len(missing)} CV fold(s) missing — see section 1")
    else:
        print("  all CV folds present")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root",
                    default=os.path.dirname(os.path.abspath(__file__)),
                    help="folder containing ablation_results/ and logs/ "
                         "(default: this script's folder)")
    ap.add_argument("--out", default=None, help="output .md path")
    ap.add_argument("--no-per-case", action="store_true",
                    help="omit section 6 (much smaller file)")
    ap.add_argument("--csv", action="store_true",
                    help="also write a flat per-fold CSV next to the summary")
    main(ap.parse_args())
