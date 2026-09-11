#!/usr/bin/env python3
"""
==============================================================
  04_baseline_deviation.py
==============================================================
  Euclidean deviation between the INTENDED electrode position
  (the SimNIBS-optimised baseline) and the position actually
  extracted from the MR image.

  Produces violin plots with the individual observations
  overlaid as a scatter, plus a statistics table.

  WHAT THIS QUANTITY IS, AND IS NOT
  ---------------------------------
  The deviation measured here is the sum of two things that
  this comparison cannot separate:

    (1) real placement deviation -- the experimenter did not
        position the electrode exactly where the optimisation
        asked;
    (2) extraction error -- the network or the coordinate
        pipeline reported the electrode in the wrong place.

  Without a per-image manual reference, a large value cannot be
  attributed to either on its own. It is therefore reported as
  "deviation from intended", never as segmentation error. The
  QC section links it to the geometric plausibility measures
  from 01 (star angle, spoke length), which is the one handle
  available for separating the two: an implausible geometry
  points at (2), a plausible one at (1).

  A companion to 03_Violinplot_euclidean_norm.py, written as a
  separate script so that the existing figures and their
  statistics are left untouched.

--------------------------------------------------------------
  HOW TO USE
      python 04_baseline_deviation.py --network b_no_attention
      python 04_baseline_deviation.py --network b_no_attention --by electrode session project
      python 04_baseline_deviation.py --input <long csv> --output-dir Figures
==============================================================
"""

import argparse
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ── Typography ──────────────────────────────────────────────────────────────
# Figures are reproduced small in print and in slides, where matplotlib's
# defaults become unreadable. Everything is set once here rather than per call
# so the whole set stays visually consistent; scale it with --font-scale.
BASE_FONT = 17.0

# Journals typically set the caption themselves and ask that the figure carry
# no title, since a title inside the image duplicates the caption and cannot be
# edited at proof stage. --no-titles suppresses every axes and figure title
# while leaving axis labels, legends and in-plot annotations intact, because
# those are part of the data display rather than the caption.
SHOW_TITLES = True


def _title(ax, text, **kw):
    """Set an axes title unless titles are suppressed."""
    if SHOW_TITLES:
        ax.set_title(text, **kw)


def _suptitle(fig, text, **kw):
    if SHOW_TITLES:
        fig.suptitle(text, **kw)


def apply_typography(scale=1.0):
    s = BASE_FONT * scale
    plt.rcParams.update({
        "font.size":            s,
        "axes.titlesize":       s * 1.20,
        "axes.labelsize":       s * 1.12,
        "xtick.labelsize":      s * 1.02,
        "ytick.labelsize":      s * 1.02,
        "legend.fontsize":      s * 0.98,
        "legend.title_fontsize": s * 1.02,
        "figure.titlesize":     s * 1.30,
        "axes.linewidth":       1.4,
        "xtick.major.width":    1.4,
        "ytick.major.width":    1.4,
        "xtick.major.size":     6,
        "ytick.major.size":     6,
        "axes.labelpad":        9,
        "axes.titlepad":        12,
        "figure.dpi":           110,
        "savefig.dpi":          300,
    })

BASELINE_SESSION = "ses-baseline"
DIMS = ["X", "Y", "Z"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_deviations(path):
    """Long table -> one row per extracted electrode, with its deviation.

    The join is on subject + electrode + condition. Condition matters: a
    subject can have more than one intended montage, and comparing an
    extracted electrode against the wrong condition's baseline would
    manufacture a large deviation out of nothing.
    """
    d = pd.read_csv(path, low_memory=False)
    d["coordinates"] = pd.to_numeric(d["coordinates"], errors="coerce")

    keep = [c for c in ("network", "star_min_angle", "star_radius_cv",
                        "spoke_min_mm", "spoke_max_mm", "star_hub_agrees",
                        "note", "baseline_residual_mm",
                        "baseline_condition_used")
            if c in d.columns]

    wide = d.pivot_table(
        index=["subject", "session", "run", "electrode", "condition"],
        columns="dimension", values="coordinates").reset_index()

    base = wide[wide.session.eq(BASELINE_SESSION)][
        ["subject", "electrode", "condition"] + DIMS]

    # `stim` (sham / active) is a property of the baseline and is dropped by
    # the pivot, which keeps only index columns and coordinate values. It is
    # therefore taken from the raw table and attached by subject and condition,
    # the same keys the baseline itself is matched on.
    if "stim" in d.columns:
        st = (d[d.session.eq(BASELINE_SESSION)][["subject", "condition", "stim"]]
              .dropna(subset=["stim"]))
        st["stim"] = st["stim"].astype(str).str.strip()
        st = st[st["stim"] != ""].drop_duplicates(["subject", "condition"])
        if len(st):
            base = base.merge(st, on=["subject", "condition"], how="left")
    ext = wide[~wide.session.eq(BASELINE_SESSION)]

    # The join MUST include `condition`. Every subject has two baselines, a
    # target montage and a control montage on the opposite side of the head,
    # and 02 records in each extracted row which of them was actually applied.
    # Joining on subject and electrode alone forms a cartesian product: each
    # electrode matches both baselines and appears twice, once against the
    # correct montage at a few millimetres and once against the contralateral
    # one at around 146 mm. That is what produced the bimodal distributions
    # with doubled row counts, and it cannot be repaired downstream because
    # both rows are equally valid-looking.
    n_bl = base.groupby(["subject", "electrode"])["condition"].nunique()
    multi = int((n_bl > 1).sum())
    if multi:
        print(f"  {multi:,} subject/electrode pair(s) have more than one "
              f"baseline condition; the join uses the condition recorded on "
              f"each extracted row to pick the right one.")

    before = len(ext)
    merged = ext.merge(base, on=["subject", "electrode", "condition"],
                       suffixes=("", "_ref"), how="inner")

    # A row count above the number of extracted rows means the join still
    # duplicated, which would silently corrupt every statistic below.
    if len(merged) > before:
        raise SystemExit(
            f"\n  The baseline join produced {len(merged):,} rows from "
            f"{before:,} extracted rows.\n"
            f"  Some subject/electrode/condition combination has more than one "
            f"baseline,\n  so each electrode was matched more than once. "
            f"Deduplicate the baseline\n  table before continuing; the "
            f"deviations would otherwise be computed\n  against several "
            f"montages at once.")

    unmatched = before - len(merged)
    if unmatched:
        miss = ext.merge(base, on=["subject", "electrode", "condition"],
                         how="left", indicator=True)
        miss = miss[miss._merge == "left_only"]
        print(f"  {unmatched:,} extracted observation(s) had no baseline for "
              f"their recorded condition and were dropped.")
        if len(miss):
            byc = miss.groupby("condition").size()
            print("    by condition: " +
                  ", ".join(f"{k} {v:,}" for k, v in byc.items()))
            print("    If a whole condition is missing here, 02 stamped a "
                  "condition for which no baseline exists.")

    for c in DIMS:
        # Signed per-axis difference. The Euclidean norm alone cannot
        # distinguish random scatter from a constant offset; the signed
        # components can, and that distinction decides whether a project's
        # deviations mean anything about placement.
        merged[f"d{c}"] = merged[c] - merged[f"{c}_ref"]
    merged["deviation_mm"] = np.sqrt(
        sum(merged[f"d{c}"] ** 2 for c in DIMS))

    # project label, taken from the baseline's exp field where available
    if "exp" in d.columns:
        proj = (d[d.session.eq(BASELINE_SESSION)][["subject", "exp"]]
                .dropna().drop_duplicates("subject"))
        merged = merged.merge(proj, on="subject", how="left")
        merged["project"] = merged["exp"].fillna("unknown")
    else:
        merged["project"] = "unknown"

    # per-image QC columns carried through from 01
    if keep:
        qc = (d[~d.session.eq(BASELINE_SESSION)]
              [["subject", "session", "run"] + keep].drop_duplicates(
                  ["subject", "session", "run"]))
        merged = merged.merge(qc, on=["subject", "session", "run"], how="left")

    coverage = dict(extracted_rows=len(ext), matched=len(merged),
                    subjects_total=ext.subject.nunique(),
                    subjects_with_baseline=base.subject.nunique())
    return merged, coverage


def filter_stim(df, wanted):
    """Restrict to baselines that came from one stimulation folder.

    `stim` records which of sham/ or active/ the SimNIBS pickle was read from.
    It is a property of the baseline rather than of the extraction, so it only
    exists for images that were matched to one; images without a baseline
    cannot be classified either way and are dropped when a filter is applied.
    """
    if not wanted:
        return df, 0
    if "stim" not in df.columns:
        print("  (no `stim` column in this table — re-run 02 to add it; "
              "--stim ignored)")
        return df, 0
    have = sorted(x for x in df["stim"].dropna().unique() if str(x).strip())
    unknown = [w for w in wanted if w not in have]
    if unknown:
        print(f"  WARNING: no rows with stim = {', '.join(unknown)}. "
              f"Present: {', '.join(have) or '(none)'}")
    before = len(df)
    out = df[df["stim"].isin(wanted)].copy()
    print(f"\n  stimulation folder filter: keeping {', '.join(wanted)}")
    print(f"  {len(out):,} of {before:,} observations retained")
    return out, before - len(out)


def filter_notes(df, exclude, keep_only_clean):
    """Drop rows whose `note` matches one of the excluded reasons.

    02_Merge... writes a `note` naming why a row is suspect --
    contralateral_baseline?, rigid_offset?, different_montage?,
    high_residual, star_failed_swap_reverted, anode_not_central,
    ambiguous_condition. Those rows are still valid
    extractions; what is doubtful is the BASELINE they are compared against, so
    they corrupt a deviation analysis while remaining perfectly good
    coordinates for anything that does not use the baseline. Excluding them
    here rather than upstream keeps that distinction.

    Matching is a substring test, because a row can carry several reasons
    joined with '+'.
    """
    if "note" not in df.columns:
        if exclude or keep_only_clean:
            print("  (no `note` column in this table — nothing to exclude; "
                  "re-run 02 to add it)")
        return df, 0
    notes = df["note"].fillna("").astype(str)
    if keep_only_clean:
        mask = notes.str.len() == 0
        dropped = int((~mask).sum())
        reasons = (notes[~mask].str.split("+").explode().value_counts()
                   if dropped else None)
    else:
        mask = ~notes.apply(lambda s: any(e in s for e in exclude))
        dropped = int((~mask).sum())
        reasons = (notes[~mask].str.split("+").explode().value_counts()
                   if dropped else None)
    if dropped:
        print(f"\n  excluded {dropped:,} of {len(df):,} electrode "
              f"observations by note:")
        for r, n in reasons.items():
            if r:
                print(f"    {r:<28}{n:>7,}")
    return df[mask].copy(), dropped


# ---------------------------------------------------------------------------
# Systematic offset
# ---------------------------------------------------------------------------

def project_bias(df):
    """Per project: mean displacement vector, its length, and the bias ratio.

    If deviations were random placement error, the mean displacement vector
    would be near zero and its length small relative to the typical deviation.
    A long mean vector means most of the deviation is ONE constant translation,
    which is the signature of a coordinate or reference mismatch rather than of
    electrodes being placed imprecisely.

        bias_ratio = |mean displacement| / median deviation

    Near 0 -> scatter around the intended position.
    Near 1 -> essentially a rigid shift.
    """
    rows = []
    for name, g in df.groupby("project", observed=True):
        mv = np.array([g[f"d{c}"].mean() for c in DIMS])
        med = float(np.median(g.deviation_mm))
        rows.append(dict(project=str(name), n=len(g),
                         mean_dX=mv[0], mean_dY=mv[1], mean_dZ=mv[2],
                         mean_vector_mm=float(np.linalg.norm(mv)),
                         median_dev_mm=med,
                         bias_ratio=float(np.linalg.norm(mv)) / med if med else np.nan))
    return pd.DataFrame(rows).sort_values("bias_ratio")


def select_projects(df, wanted, excluded, max_ratio, bias):
    """Apply --projects / --exclude-projects / --max-bias-ratio in that order."""
    keep = set(df.project.unique())
    note = []
    if wanted:
        keep &= set(wanted)
        note.append(f"--projects {' '.join(wanted)}")
    if excluded:
        keep -= set(excluded)
        note.append(f"--exclude-projects {' '.join(excluded)}")
    if max_ratio is not None:
        low = set(bias[bias.bias_ratio <= max_ratio].project)
        dropped = sorted(keep - low)
        keep &= low
        note.append(f"--max-bias-ratio {max_ratio:g}"
                    + (f" (dropped {', '.join(dropped)})" if dropped else ""))
    return df[df.project.isin(keep)].copy(), sorted(keep), "; ".join(note)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def describe(df, group):
    """Descriptives per group. Median and IQR lead, because the deviation
    distribution is right-skewed and the mean is pulled by a small number of
    large values."""
    rows = []
    for name, g in df.groupby(group, observed=True):
        v = g.deviation_mm.dropna().values
        if len(v) == 0:
            continue
        rows.append(dict(
            group=str(name), n=len(v),
            mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else np.nan,
            median=np.median(v),
            q25=np.percentile(v, 25), q75=np.percentile(v, 75),
            p95=np.percentile(v, 95), max=v.max(),
            pct_gt_5mm=(v > 5).mean() * 100,
            pct_gt_10mm=(v > 10).mean() * 100,
            pct_gt_20mm=(v > 20).mean() * 100))
    return pd.DataFrame(rows).sort_values("median")


def compare_groups(df, group, alpha=0.05):
    """Kruskal-Wallis across groups, then Holm-corrected pairwise tests.

    Non-parametric throughout: the deviations are bounded below by zero and
    right-skewed, so normality is not a reasonable assumption. Holm rather
    than Bonferroni because it is uniformly more powerful at the same
    family-wise error rate.
    """
    groups = [g.deviation_mm.dropna().values
              for _, g in df.groupby(group, observed=True)]
    names = [str(n) for n, _ in df.groupby(group, observed=True)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return None, pd.DataFrame()

    H, p = stats.kruskal(*groups)
    pairs = []
    for (i, a), (j, b) in itertools.combinations(enumerate(groups), 2):
        u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
        # rank-biserial correlation: an effect size that goes with the test
        r = 1 - (2 * u) / (len(a) * len(b))
        pairs.append(dict(a=names[i], b=names[j], n_a=len(a), n_b=len(b),
                          median_a=np.median(a), median_b=np.median(b),
                          U=u, p_raw=pu, effect_r=abs(r)))
    pd_pairs = pd.DataFrame(pairs)
    if not pd_pairs.empty:
        order = pd_pairs.p_raw.values.argsort()
        m = len(pd_pairs)
        adj = np.empty(m)
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (m - rank) * pd_pairs.p_raw.values[idx])
            adj[idx] = min(1.0, running)
        pd_pairs["p_holm"] = adj
        pd_pairs["significant"] = pd_pairs.p_holm < alpha
        pd_pairs = pd_pairs.sort_values("p_holm")
    return dict(H=H, p=p, k=len(groups)), pd_pairs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def violin_scatter(df, group, out_path, title, max_points=4000):
    """Violin showing the distribution, with the observations on top.

    The scatter is the point of this figure: a violin alone smooths the data
    into a shape that looks equally trustworthy whether it rests on 20 points
    or 2000. Overlaying the observations shows how many there are and where
    they actually sit.
    """
    order = (df.groupby(group, observed=True).deviation_mm
             .median().sort_values().index.tolist())
    n_g = len(order)
    fig, ax = plt.subplots(figsize=(max(8, 1.7 * n_g + 3), 7))

    sns.violinplot(data=df, x=group, y="deviation_mm", order=order,
                   inner=None, cut=0, linewidth=1.0, density_norm="width",
                   color="#cfd8e3", ax=ax)
    plot_df = df
    if len(df) > max_points:                     # keep the figure readable
        plot_df = df.sample(max_points, random_state=0)
    sns.stripplot(data=plot_df, x=group, y="deviation_mm", order=order,
                  size=3.2, alpha=0.40, jitter=0.28, color="#22384f", ax=ax)
    sns.boxplot(data=df, x=group, y="deviation_mm", order=order, width=0.12,
                showcaps=False, showfliers=False, boxprops=dict(alpha=0.9),
                whiskerprops=dict(linewidth=1.0),
                medianprops=dict(color="#c62828", linewidth=2), ax=ax)

    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.20)
    for i, g in enumerate(order):
        v = df[df[group] == g].deviation_mm
        ax.text(i, ax.get_ylim()[1] * 0.985,
                f"n={len(v)}\nmdn {np.median(v):.1f}",
                ha="center", va="top",
                fontsize=plt.rcParams["font.size"] * 0.96,
                color="#33475b", fontweight="medium")

    ax.set_xlabel("")
    ax.set_ylabel("Deviation from intended position (mm)")
    _title(ax, title)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(width=1.1, length=5)
    if len(str(order[0])) > 8:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def planned_to_actual(df, out_path, network, max_lines=900, by_electrode=True):
    """Three orthogonal projections, each drawing a segment from the intended
    position to the extracted one.

    A violin summarises how FAR the electrodes are from where they should be;
    this shows in WHICH DIRECTION. If the segments fan out evenly the deviation
    is scatter; if they all point the same way the projection is offset, and no
    amount of distance summary would have revealed that.
    """
    planes = [("X", "Y"), ("X", "Z"), ("Y", "Z")]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.4))

    sub = df if len(df) <= max_lines else df.sample(max_lines, random_state=0)
    electrodes = sorted(df.electrode.unique())
    palette = dict(zip(electrodes,
                       sns.color_palette("colorblind", len(electrodes))))

    for ax, (h, v) in zip(axes, planes):
        for _, r in sub.iterrows():
            col = palette[r.electrode] if by_electrode else "#22384f"
            ax.plot([r[f"{h}_ref"], r[h]], [r[f"{v}_ref"], r[v]],
                    color=col, alpha=0.35, linewidth=0.9, zorder=1)
        ax.scatter(sub[f"{h}_ref"], sub[f"{v}_ref"], s=16, alpha=0.60,
                   facecolor="none", edgecolor="#444444", linewidth=0.9,
                   zorder=2, label="intended")
        ax.scatter(sub[h], sub[v], s=13, alpha=0.60,
                   color=[palette[e] for e in sub.electrode],
                   zorder=3, label="extracted")
        ax.set_xlabel(f"{h} (mm)")
        ax.set_ylabel(f"{v} (mm)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        ax.tick_params(width=1.1, length=5)

    handles = [plt.Line2D([], [], color=palette[e], marker="o", linestyle="",
                          markersize=9, label=e) for e in electrodes]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                              markerfacecolor="none", markeredgecolor="#444444",
                              label="intended"))
    axes[-1].legend(handles=handles, loc="best", framealpha=0.92,
                    borderpad=0.7, labelspacing=0.55)
    _suptitle(fig, f"Intended vs. extracted electrode positions — {network}"
                   f"   ({len(sub)} of {len(df)} shown)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def displacement_cloud(df, out_path, network):
    """Displacement vectors moved to a common origin, one panel per plane.

    Centring every vector at (0,0) makes a systematic offset unmissable: random
    placement error gives a cloud centred on the origin, a coordinate mismatch
    gives a cloud displaced away from it. The cross marks the mean.
    """
    planes = [("X", "Y"), ("X", "Z"), ("Y", "Z")]
    # One symmetric limit shared by every axis of every panel. Letting each
    # panel autoscale makes a 5 mm spread and a 30 mm spread look identical,
    # which defeats the purpose of the figure: the point is to compare the
    # spread BETWEEN axes, and that comparison is only readable on a common
    # scale.
    lim = float(np.nanmax([np.abs(df[f"d{c}"]).max() for c in DIMS]))
    lim = np.ceil(lim / 5.0) * 5.0

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.4))
    for ax, (h, v) in zip(axes, planes):
        x, y = df[f"d{h}"], df[f"d{v}"]
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.axvline(0, color="#999999", linewidth=0.8)
        ax.scatter(x, y, s=11, alpha=0.30, color="#22384f")
        ax.plot(x.mean(), y.mean(), "x", color="#c62828", markersize=20,
                markeredgewidth=3.5, zorder=5)
        ax.set_xlabel(f"\u0394{h} (mm)")
        ax.set_ylabel(f"\u0394{v} (mm)")
        _title(ax, f"mean ({x.mean():+.1f}, {y.mean():+.1f}) mm")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.tick_params(width=1.1, length=5)
    mv = np.linalg.norm([df[f"d{c}"].mean() for c in DIMS])
    _suptitle(fig, f"Displacement vectors at a common origin — {network}   "
                   f"|mean| = {mv:.2f} mm, median deviation = "
                   f"{np.median(df.deviation_mm):.2f} mm")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _half_violin(ax, values, pos, side, color, width=0.34):
    """One half of a violin, drawn from a KDE, facing `side`."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return
    kde = stats.gaussian_kde(v)
    grid = np.linspace(v.min(), v.max(), 200)
    dens = kde(grid)
    dens = dens / dens.max() * width
    off = dens if side == "right" else -dens
    ax.fill_betweenx(grid, pos, pos + off, color=color, alpha=0.45, lw=0)
    ax.plot(pos + off, grid, color=color, lw=1.0, alpha=0.9)


def raincloud_paired(df, out_path, network, metric, title, ylabel,
                     max_points=1500):
    """Intended vs extracted for one quantity, as a raincloud.

    Half violin for the distribution, jittered scatter for the observations,
    and a slim box for the quartiles. The scatter is coloured by project
    because the projects behave differently and a single pooled cloud would
    hide that.

    The two distributions are PAIRED -- every extracted electrode has exactly
    one intended counterpart -- so the test is a Wilcoxon signed-rank on the
    differences, not a comparison of two independent samples. Plotting them
    side by side makes them look independent, which is why the statistics
    box states the paired result explicitly.
    """
    if metric == "norm":
        a = np.sqrt(sum(df[f"{c}_ref"] ** 2 for c in DIMS))
        b = np.sqrt(sum(df[c] ** 2 for c in DIMS))
    else:
        a, b = df[f"{metric}_ref"], df[metric]
    ok_mask = np.isfinite(a) & np.isfinite(b)
    a, b = np.asarray(a[ok_mask]), np.asarray(b[ok_mask])
    proj = df.loc[ok_mask, "project"].values

    fig, ax = plt.subplots(figsize=(9.5, 7.8))
    projects = sorted(pd.unique(proj))
    palette = dict(zip(projects, sns.color_palette("colorblind", len(projects))))

    rng = np.random.default_rng(0)
    idx = np.arange(len(a))
    if len(idx) > max_points:
        idx = rng.choice(idx, max_points, replace=False)

    for pos, vals, side, base in ((0, a, "left", "#8d99ae"),
                                  (1, b, "right", "#8d99ae")):
        _half_violin(ax, vals, pos, side, base)
        jit = rng.uniform(0.06, 0.30, len(idx)) * (1 if side == "left" else -1)
        ax.scatter(pos + jit, vals[idx], s=13, alpha=0.50,
                   color=[palette[p] for p in proj[idx]], linewidths=0, zorder=3)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.add_patch(plt.Rectangle((pos - 0.045, q1), 0.09, q3 - q1,
                                   facecolor="white", edgecolor="#33475b",
                                   lw=1.1, zorder=4))
        ax.plot([pos - 0.06, pos + 0.06], [med, med], color="#c62828",
                lw=3.0, zorder=5)

    # paired statistics
    d = b - a
    try:
        w, pw = stats.wilcoxon(a, b)
    except ValueError:
        w, pw = np.nan, np.nan
    se = d.std(ddof=1) / np.sqrt(len(d))
    lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
    txt = (f"n = {len(d):,} paired\n"
           f"median intended  {np.median(a):8.2f}\n"
           f"median extracted {np.median(b):8.2f}\n"
           f"mean difference  {d.mean():+8.2f}\n"
           f"   95% CI [{lo:+.2f}, {hi:+.2f}]\n"
           f"Wilcoxon signed-rank\n"
           f"   W = {w:.0f}, p = {pw:.3g}")
    ax.text(0.985, 0.015, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=plt.rcParams["font.size"] * 0.90, family="monospace",
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f6f8",
                      edgecolor="#c9ced6", linewidth=1.1))

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                          color=palette[p], label=p) for p in projects]
    ax.legend(handles=handles, title="Project", loc="upper left",
              framealpha=0.92, ncol=1 if len(projects) < 6 else 2,
              markerscale=1.0, borderpad=0.7, labelspacing=0.6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Intended\n(planned)", "Extracted\n(network)"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel(ylabel)
    _title(ax, f"{title} — {network}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def dimension_violin(df, out_path, network, max_points=2500,
                     scatter=True, annotate=True):
    """Signed deviation per axis, pooled over sessions.

    The Euclidean norm answers "how far", which is always positive and
    therefore cannot show direction. Splitting it into the signed components
    does: a distribution centred on zero is scatter, one displaced from zero is
    a bias on that axis. Sessions are pooled deliberately -- the session
    figure already shows there is no meaningful drift between them, so keeping
    them apart here would only divide the sample.
    """
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    rng = np.random.default_rng(0)
    cols = [f"d{c}" for c in DIMS]
    data = [df[c].dropna().values for c in cols]

    ax.axhline(0, color="#c62828", lw=1.6, ls="--", zorder=1,
               label="no deviation")

    parts = ax.violinplot(data, positions=range(len(DIMS)), widths=0.75,
                          showextrema=False, showmedians=False)
    for b in parts["bodies"]:
        b.set_facecolor("#cfd8e3")
        b.set_alpha(0.65)
        b.set_edgecolor("#7d8894")
        b.set_linewidth(1.0)

    projects = sorted(df.project.unique())
    palette = dict(zip(projects, sns.color_palette("colorblind", len(projects))))
    for i, c in enumerate(cols):
        if scatter:
            sub = df[[c, "project"]].dropna()
            if len(sub) > max_points:
                sub = sub.sample(max_points, random_state=0)
            jit = rng.uniform(-0.20, 0.20, len(sub))
            ax.scatter(i + jit, sub[c], s=9, alpha=0.35, linewidths=0,
                       color=[palette[p] for p in sub.project], zorder=3)
        v = data[i]
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.add_patch(plt.Rectangle((i - 0.055, q1), 0.11, q3 - q1,
                                   facecolor="white", edgecolor="#33475b",
                                   lw=1.2, zorder=4))
        ax.plot([i - 0.075, i + 0.075], [med, med], color="#c62828", lw=3,
                zorder=5)

    if annotate:
        # Four lines of text at this size need room; extend the axis rather
        # than letting the annotation sit on top of the distributions.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.26)
        top = ax.get_ylim()[1]
        for i, v in enumerate(data):
            try:
                _, pv = stats.wilcoxon(v)
                ptxt = f"p = {pv:.2g}"
            except ValueError:
                ptxt = ""
            ax.text(i, top * 0.985,
                    f"n = {len(v):,}\nmedian {np.median(v):+.2f} mm\n"
                    f"mean {v.mean():+.2f} mm\n{ptxt}",
                    ha="center", va="top",
                    fontsize=plt.rcParams["font.size"] * 0.94, color="#33475b")

    if scatter:
        handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                              color=palette[p], label=p) for p in projects]
        ax.legend(handles=handles, title="Project", loc="lower left",
                  framealpha=0.92, borderpad=0.6, labelspacing=0.5,
                  ncol=1 if len(projects) < 6 else 2)

    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels([f"$\\Delta${c}" for c in DIMS])
    ax.set_ylabel("Extracted − intended (mm)")
    _title(ax, f"Signed deviation per axis, all sessions pooled — {network}")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(width=1.1, length=5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def dimension_bars(df, out_path, network, errorbar="sd"):
    """Mean signed deviation per axis with an error bar.

    The default error bar is the standard deviation, which describes how much
    individual electrodes vary. That is not the same as the uncertainty of the
    mean: with n in the thousands the standard error is roughly thirty times
    smaller, and a bar drawn with SEM would look decisively different from zero
    even for a shift of a few tenths of a millimetre. Pass errorbar="sem" if
    the uncertainty of the estimate is what is wanted, but say which was used.
    """
    cols = [f"d{c}" for c in DIMS]
    data = [df[c].dropna().values for c in cols]
    means = [v.mean() for v in data]
    sds = [v.std(ddof=1) for v in data]
    errs = (sds if errorbar == "sd"
            else [s / np.sqrt(len(v)) for s, v in zip(sds, data)])

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axhline(0, color="#c62828", lw=1.6, ls="--", zorder=1)
    ax.bar(range(len(DIMS)), means, yerr=errs, width=0.55,
           color="#5b7fa6", edgecolor="#28425e", linewidth=1.4,
           error_kw=dict(ecolor="#22384f", capsize=9, capthick=1.8,
                         elinewidth=1.8), zorder=3)
    # Extend the axis before labelling, so the values sit inside the plot area
    # rather than being clipped at the frame or colliding with the title.
    span = max(abs(np.array(means)) + np.array(errs))
    ax.set_ylim(-span * 1.32, span * 1.32)
    for i, (m, e) in enumerate(zip(means, errs)):
        y = m + (e + span * 0.07) * (1 if m >= 0 else -1)
        ax.text(i, y, f"{m:+.2f}", ha="center",
                va="bottom" if m >= 0 else "top",
                fontsize=plt.rcParams["font.size"] * 0.96,
                color="#22384f", fontweight="medium")

    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels([f"$\\Delta${c}" for c in DIMS])
    lab = "SD" if errorbar == "sd" else "SEM"
    ax.set_ylabel(f"Extracted − intended (mm), mean ± {lab}")
    _title(ax, f"Mean signed deviation per axis — {network}")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(width=1.4, length=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def qc_scatter(df, out_path, network):
    """Deviation against geometric plausibility, when 01's columns are present.

    If large deviations sit mostly at low star angles, they are extraction
    failures. If they are spread across plausible geometries, they are more
    likely real placement deviation. This plot does not decide that, but it is
    what makes the question answerable.
    """
    if "star_min_angle" not in df.columns or df.star_min_angle.isna().all():
        return None
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, col, lab in ((axes[0], "star_min_angle",
                          "Minimum inter-spoke angle (deg)"),
                         (axes[1], "spoke_max_mm", "Longest spoke (mm)")):
        if col not in df.columns:
            ax.axis("off")
            continue
        sub = df.dropna(subset=[col, "deviation_mm"])
        ax.scatter(sub[col], sub.deviation_mm, s=11, alpha=0.30, color="#22384f")
        ax.set_xlabel(lab)
        ax.set_ylabel("Deviation from intended (mm)")
        ax.grid(alpha=0.25)
        ax.tick_params(width=1.1, length=5)
        if len(sub) > 2:
            rho, p = stats.spearmanr(sub[col], sub.deviation_mm)
            _title(ax, f"Spearman $\\rho$ = {rho:+.3f}   (p = {p:.2g})")
    _suptitle(fig, f"Deviation vs. geometric plausibility — {network}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(a):
    global SHOW_TITLES
    SHOW_TITLES = not a.no_titles
    apply_typography(a.font_scale)
    here = Path(__file__).resolve().parent
    tables = Path(a.tables) if a.tables else here / "Tables"
    outdir = Path(a.output_dir) if a.output_dir else here / "Figures"
    outdir.mkdir(parents=True, exist_ok=True)

    src = Path(a.input) if a.input else (
        tables / f"corrected_electrode_positions_with_baseline_long_{a.network}.csv")
    if not src.exists():
        raise SystemExit(
            f"Input not found: {src}\n"
            f"Run: python 02_Merge_tables_and_Corret_Naming.py --network {a.network}")

    # The electrode selection goes into the filename as well as the title: a
    # figure showing only the anode must not be mistaken for the whole montage.
    el_tag = ("_" + "-".join(a.electrodes)) if a.electrodes else ""
    stamp = f"{a.network}{el_tag}_{datetime.now():%Y%m%d}"
    print(f"\n{'=' * 62}")
    print("  Deviation from intended electrode position")
    print(f"{'=' * 62}")
    print(f"  Network : {a.network}")
    print(f"  Input   : {src}")
    print(f"  Output  : {outdir}")
    print(f"{'=' * 62}\n")

    df, cov = load_deviations(src)
    print(f"  extracted electrode observations : {cov['extracted_rows']:,}")
    print(f"  matched to an intended position  : {cov['matched']:,} "
          f"({cov['matched'] / max(cov['extracted_rows'], 1) * 100:.1f}%)")
    print(f"  subjects with a baseline         : "
          f"{cov['subjects_with_baseline']} of {cov['subjects_total']}")
    if cov['matched'] < cov['extracted_rows'] * 0.5:
        print("\n  NOTE: fewer than half of the extracted electrodes have an")
        print("        intended position to compare against. Everything below")
        print("        describes that subset, not the whole cohort.")
    if df.empty:
        raise SystemExit("\nNo electrode could be matched to a baseline.")

    v = df.deviation_mm
    print(f"\n  ── overall ({len(v):,} electrodes) ──")
    print(f"  median {np.median(v):6.2f} mm   IQR "
          f"[{np.percentile(v,25):.2f}, {np.percentile(v,75):.2f}]")
    print(f"  mean   {v.mean():6.2f} mm   SD {v.std(ddof=1):.2f}")
    print(f"  p95    {np.percentile(v,95):6.2f} mm   max {v.max():.2f}")
    for t in (5, 10, 20):
        print(f"  > {t:2d} mm : {(v > t).mean() * 100:5.2f}%")

    if a.list_electrodes:
        print("  electrodes present:")
        for e, n in df.electrode.value_counts().sort_index().items():
            print(f"    {e:<12} {n:>6,} observations")
        print()
        raise SystemExit(0)

    if a.electrodes:
        known = sorted(df.electrode.unique())
        unknown = [e for e in a.electrodes if e not in known]
        if unknown:
            raise SystemExit(
                f"\nUnknown electrode(s): {', '.join(unknown)}\n"
                f"Available: {', '.join(known)}")
        before = len(df)
        df = df[df.electrode.isin(a.electrodes)].copy()
        print(f"\n  electrode selection: {', '.join(a.electrodes)}")
        print(f"  {len(df):,} of {before:,} observations retained")
        if df.empty:
            raise SystemExit("\nNo observations left after electrode selection.")
        # A single electrode makes the per-electrode grouping degenerate, so
        # drop it rather than emitting a one-category violin.
        if len(a.electrodes) == 1 and "electrode" in a.by:
            a.by = [g for g in a.by if g != "electrode"]
            print("  ('electrode' dropped from --by: only one category remains)")
        v = df.deviation_mm
        print(f"\n  ── selected electrode(s) ({len(v):,} observations) ──")
        print(f"  median {np.median(v):6.2f} mm   IQR "
              f"[{np.percentile(v,25):.2f}, {np.percentile(v,75):.2f}]")
        print(f"  mean   {v.mean():6.2f} mm   SD {v.std(ddof=1):.2f}")

    if a.stim:
        df, _ = filter_stim(df, a.stim)
        if df.empty:
            raise SystemExit("\nNo observations left after the stim filter.")

    if a.exclude_note or a.clean_only:
        df, _ = filter_notes(df, a.exclude_note or [], a.clean_only)
        if df.empty:
            raise SystemExit("\nNo observations left after note filtering.")
        v = df.deviation_mm
        print(f"\n  ── after note filtering ({len(v):,} observations) ──")
        print(f"  median {np.median(v):6.2f} mm   IQR "
              f"[{np.percentile(v,25):.2f}, {np.percentile(v,75):.2f}]")

    bias = project_bias(df)
    print(f"\n  ── systematic offset per project ──")
    print("  bias_ratio = |mean displacement| / median deviation;")
    print("  low = scatter around the intended position, high = a rigid shift.")
    print(bias.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    bias.to_csv(outdir / f"project_bias_{stamp}.csv", index=False)

    df_all = df
    if a.projects or a.exclude_projects or a.max_bias_ratio is not None:
        df, kept, note = select_projects(df, a.projects, a.exclude_projects,
                                         a.max_bias_ratio, bias)
        print(f"\n  project selection: {note}")
        print(f"  keeping {len(kept)} project(s): {', '.join(kept)}")
        print(f"  {len(df):,} of {len(df_all):,} electrodes retained")
        if df.empty:
            raise SystemExit("\nNo electrodes left after project selection.")
        v = df.deviation_mm
        print(f"\n  ── selected subset ({len(v):,} electrodes) ──")
        print(f"  median {np.median(v):6.2f} mm   IQR "
              f"[{np.percentile(v,25):.2f}, {np.percentile(v,75):.2f}]")
        print(f"  mean   {v.mean():6.2f} mm   SD {v.std(ddof=1):.2f}")
        mv = np.linalg.norm([df[f"d{c}"].mean() for c in DIMS])
        print(f"  |mean displacement| {mv:.2f} mm "
              f"(ratio {mv/np.median(v):.2f})")

    written = []
    for group in a.by:
        if group not in df.columns:
            print(f"\n  (skipping '{group}': not a column)")
            continue
        desc = describe(df, group)
        kw, pairs = compare_groups(df, group)

        print(f"\n  ── by {group} ──")
        print(desc.to_string(index=False,
                             float_format=lambda x: f"{x:.2f}"))
        if kw:
            print(f"\n  Kruskal-Wallis across {kw['k']} groups: "
                  f"H = {kw['H']:.2f}, p = {kw['p']:.3g}")
            sig = pairs[pairs.significant] if "significant" in pairs else pairs
            if len(sig):
                print(f"  {len(sig)} of {len(pairs)} pairwise contrasts "
                      f"significant after Holm correction; strongest:")
                print(sig.head(5)[["a", "b", "median_a", "median_b",
                                   "p_holm", "effect_r"]].to_string(
                    index=False, float_format=lambda x: f"{x:.3g}"))
            else:
                print("  no pairwise contrast survives Holm correction")

        desc.to_csv(outdir / f"deviation_stats_{group}_{stamp}.csv", index=False)
        if not pairs.empty:
            pairs.to_csv(outdir / f"deviation_pairwise_{group}_{stamp}.csv",
                         index=False)
        p = violin_scatter(
            df, group, outdir / f"deviation_violin_{group}_{stamp}.png",
            f"Deviation from intended position by {group} — {a.network}")
        written.append(p)

    written.append(raincloud_paired(
        df, outdir / f"raincloud_norm_{stamp}.png", a.network, "norm",
        "Euclidean norm of the position vector",
        r"$\sqrt{X^2+Y^2+Z^2}$ (mm)"))
    for dim in DIMS:
        written.append(raincloud_paired(
            df, outdir / f"raincloud_{dim}_{stamp}.png", a.network, dim,
            f"{dim} coordinate", f"{dim} (mm)"))

    written.append(dimension_violin(
        df, outdir / f"deviation_by_dimension_{stamp}.png", a.network))
    written.append(dimension_violin(
        df, outdir / f"deviation_by_dimension_plain_{stamp}.png", a.network,
        scatter=False, annotate=False))
    written.append(dimension_bars(
        df, outdir / f"deviation_by_dimension_bars_{stamp}.png", a.network,
        errorbar=a.errorbar))

    written.append(planned_to_actual(
        df, outdir / f"planned_vs_actual_{stamp}.png", a.network,
        max_lines=a.max_lines))
    written.append(displacement_cloud(
        df, outdir / f"displacement_cloud_{stamp}.png", a.network))

    q = qc_scatter(df, outdir / f"deviation_vs_geometry_{stamp}.png", a.network)
    if q:
        written.append(q)

    df.to_csv(outdir / f"deviation_per_electrode_{stamp}.csv", index=False)

    print(f"\n  ── written ──")
    for w in written:
        print(f"    {Path(w).name}")
    print(f"    deviation_per_electrode_{stamp}.csv")
    print(f"\n{'=' * 62}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default="b_no_attention")
    ap.add_argument("--input", default=None,
                    help="explicit long CSV, overriding the --network lookup")
    ap.add_argument("--tables", default=None)
    ap.add_argument("--output-dir", dest="output_dir", default=None)
    ap.add_argument("--stim", nargs="+", default=None,
                    choices=["sham", "active"], metavar="FOLDER",
                    help="keep only baselines read from these stimulation "
                         "folders, e.g. --stim active")
    ap.add_argument("--exclude-note", dest="exclude_note", nargs="+",
                    default=None, metavar="REASON",
                    help="drop observations whose `note` contains any of these, "
                         "e.g. --exclude-note contralateral_baseline? "
                         "high_residual. Matching is a substring test.")
    ap.add_argument("--clean-only", dest="clean_only", action="store_true",
                    help="keep only observations with an empty `note`")
    ap.add_argument("--electrodes", nargs="+", default=None,
                    metavar="NAME",
                    help="restrict the analysis to these electrodes, e.g. "
                         "--electrodes anode. Useful because the anode is the "
                         "central reference of the montage and behaves "
                         "differently from the three cathodes. Use --list-electrodes "
                         "to see the available names.")
    ap.add_argument("--list-electrodes", dest="list_electrodes",
                    action="store_true",
                    help="print the electrode names present in the table, then exit")
    ap.add_argument("--projects", nargs="+", default=None,
                    help="keep only these projects, e.g. --projects P1 P4")
    ap.add_argument("--exclude-projects", dest="exclude_projects", nargs="+",
                    default=None, help="drop these projects")
    ap.add_argument("--max-bias-ratio", dest="max_bias_ratio", type=float,
                    default=None,
                    help="keep only projects whose |mean displacement| divided "
                         "by their median deviation is at or below this. Use it "
                         "to restrict the analysis to projects showing no "
                         "systematic offset, e.g. --max-bias-ratio 0.4")
    ap.add_argument("--errorbar", choices=["sd", "sem"], default="sem",
                    help="error bar on the bar chart: sd describes the spread "
                         "between electrodes, sem the uncertainty of the mean. "
                         "They differ by a factor of sqrt(n) and answer "
                         "different questions.")
    ap.add_argument("--no-titles", dest="no_titles", action="store_true",
                    help="omit all figure and panel titles, for journals that "
                         "set the caption separately. Axis labels, legends and "
                         "in-plot statistics are kept.")
    ap.add_argument("--font-scale", dest="font_scale", type=float, default=1.0,
                    help="multiply every font size and line width. The default "
                         "already suits a full-width figure; raise it to about "
                         "1.3 for slides or a two-column layout where the "
                         "figure is reproduced small, or lower it to 0.8 to "
                         "recover the previous appearance.")
    ap.add_argument("--max-lines", dest="max_lines", type=int, default=900,
                    help="segments drawn in the planned-vs-actual figure")
    ap.add_argument("--by", nargs="+",
                    default=["electrode", "session", "project"],
                    help="grouping variables to plot and test")
    main(ap.parse_args())
