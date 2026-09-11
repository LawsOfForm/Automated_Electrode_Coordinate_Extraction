"""
==============================================================
  03_Violinplot_euclidean_norm.py
==============================================================
  Author  : Filip Niemann
  Contact : filip.niemann@med.uni-greifswald.de
  Questions, bug reports, and feature requests are welcome —
  please reach out by e-mail.
--------------------------------------------------------------
  DESCRIPTION
  -----------
  Creates two violin-plot figures from the corrected electrode
  coordinate table produced by 02_Merge_tables_2.py:

    Figure 1 — Euclidean norm by session (all subjects, all
               runs, all conditions pooled).

    Figure 2 — Same data, but violin colour encodes the
               study project derived from the subject ID:

               Subject ID logic
               ────────────────
               sub-XXXX (4 digits) → project = first digit
                 sub-1xxx → Project 1
                 sub-2xxx → Project 2  … etc.
               sub-XXX  (3 digits) → labelled "Sham"
               sub-XXXXX (5+ digits) or non-numeric → dropped

  Both figures are saved as high-resolution PNG files next to
  this script.
--------------------------------------------------------------
  HOW TO USE
  ----------
  Edit input_csv and output_dir at the bottom of this file,
  then run:
      python 03_Violinplot_euclidean_norm.py
==============================================================
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Styling ───────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE_PROJECT = {
    "Project 1":  "#4C72B0",
    "Project 2":  "#DD8452",
    "Project 3":  "#55A868",
    "Project 4":  "#C44E52",
    "Project 5":  "#8172B2",
    "Project 6":  "#937860",
    "Project 7":  "#DA8BC3",
    "Project 8":  "#8C8C8C",
    "Sham":       "#CCB974",
}

SESSION_ORDER  = ["ses-1", "ses-2", "ses-3", "ses-4", "ses-baseline"]
SESSION_LABELS = {
    "ses-1":        "Session 1",
    "ses-2":        "Session 2",
    "ses-3":        "Session 3",
    "ses-4":        "Session 4",
    "ses-baseline": "Baseline",
}

# Figure 1 — electrode constants
ELEC_ORDER   = ["anode", "cathode1", "cathode2", "cathode3"]
ELEC_LABELS  = {
    "anode":    "Anode",
    "cathode1": "Cathode 1",
    "cathode2": "Cathode 2",
    "cathode3": "Cathode 3",
}
ELEC_PALETTE = {
    "Anode":     "#E05C5C",
    "Cathode 1": "#4C72B0",
    "Cathode 2": "#55A868",
    "Cathode 3": "#DD8452",
}
SESSION_GROUP_ORDER = ["Session 1–4", "Baseline"]


# ── Project assignment ─────────────────────────────────────────────────────────

def assign_project(subject: str) -> str | None:
    """
    Return project label for a subject ID string.

    Rules
    -----
    sub-XXXX (exactly 4 digits) → "Project <first digit>"
    sub-XXX  (exactly 3 digits) → "Sham"
    anything else (5+ digits, non-numeric, sub-CO, …) → None  (row dropped)
    """
    m = re.match(r'^sub-(\d+)$', subject)
    if not m:
        return None                          # non-numeric (e.g. sub-CO)
    digits = m.group(1)
    if len(digits) == 3:
        return "Sham"
    if len(digits) == 4:
        return f"Project {digits[0]}"
    return None                              # 5+ digits — drop


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'project' column and drop rows with no valid project."""
    df = df.copy()
    df["project"] = df["subject"].apply(assign_project)

    n_before = len(df)
    df = df.dropna(subset=["project", "euclidean_norm"])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  ℹ  Dropped {n_dropped} rows (no valid project or missing norm).")

    # Replace raw session strings with display labels
    df["session_label"] = df["session"].map(SESSION_LABELS).fillna(df["session"])
    return df


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _session_label_order(df: pd.DataFrame) -> list[str]:
    """Return session labels in the canonical order, keeping only those present."""
    present = set(df["session_label"].unique())
    return [SESSION_LABELS[s] for s in SESSION_ORDER if SESSION_LABELS[s] in present]


def _strip_inner_points(ax):
    """Remove individual strip-plot points added as overlay."""
    pass   # placeholder if we add jitter later


def _add_n_annotations(ax, df, x_col, session_order):
    """Print N= beneath each x-tick."""
    counts = df.groupby(x_col)["euclidean_norm"].count()
    for i, ses in enumerate(session_order):
        n = counts.get(ses, 0)
        ax.text(i, ax.get_ylim()[0] - 1.5, f"n={n}",
                ha="center", va="top", fontsize=9, color="dimgray")


# ── Figure 1 — session only ────────────────────────────────────────────────────

def _permutation_test(g1: np.ndarray, g2: np.ndarray,
                      n_perm: int = 10_000, seed: int = 42) -> tuple[float, float]:
    """
    Two-sided permutation test of the difference in means.

    Returns
    -------
    obs_diff : float   mean(g1) − mean(g2)
    p_value  : float   two-tailed permutation p-value
    """
    rng      = np.random.default_rng(seed)
    obs_diff = g1.mean() - g2.mean()
    combined = np.concatenate([g1, g2])
    n1       = len(g1)

    # Vectorised shuffling for speed
    idx = np.stack([rng.permutation(len(combined)) for _ in range(n_perm)])
    shuffled = combined[idx]
    null_diffs = shuffled[:, :n1].mean(axis=1) - shuffled[:, n1:].mean(axis=1)
    p = float(np.mean(np.abs(null_diffs) >= np.abs(obs_diff)))
    # Guard against p=0 (resolution floor)
    p = max(p, 1.0 / n_perm)
    return obs_diff, p


def _sig_label(p: float) -> str:
    """Convert p-value to significance stars."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _draw_sig_bracket(ax, x1: float, x2: float, y: float, label: str,
                      color: str = "black", lw: float = 1.2,
                      tip_len: float = 0.5):
    """Draw a bracket between x1 and x2 at height y with a significance label."""
    ax.plot([x1, x1, x2, x2],
            [y, y + tip_len, y + tip_len, y],
            color=color, linewidth=lw, clip_on=False)
    ax.text((x1 + x2) / 2, y + tip_len * 1.1,
            label, ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=color)



# ── Outlier detection & log ────────────────────────────────────────────────────

def write_outlier_log(df_full: pd.DataFrame, log_path: str,
                      iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers in euclidean_norm using the Tukey IQR fence method,
    computed separately for every (electrode × session_group) combination.

    A row is flagged as an outlier when:
        norm  <  Q1 − iqr_multiplier × IQR   (lower fence)
        norm  >  Q3 + iqr_multiplier × IQR   (upper fence)

    The log file contains:
      • A summary table: n outliers per electrode / session group
      • Per-project breakdown
      • A full sorted detail table of every outlier row
      • A per-subject summary listing which subjects appear most often

    Parameters
    ----------
    df_full         : prepared DataFrame (must contain 'project', 'session_group')
    log_path        : path to write the .txt log file
    iqr_multiplier  : fence multiplier (default 1.5 = Tukey standard)

    Returns
    -------
    DataFrame of outlier rows (for optional further analysis)
    """
    df = df_full.copy()
    # Add session_group if not already present
    if "session_group" not in df.columns:
        df["session_group"] = df["session"].apply(
            lambda s: "Baseline"    if s == "ses-baseline" else
                      "Session 1–4" if s in ["ses-1", "ses-2", "ses-3", "ses-4"] else None
        )
    df = df.dropna(subset=["euclidean_norm", "session_group"])

    # ── Compute per-group fences ──────────────────────────────────────────────
    grp = df.groupby(["electrode", "session_group"])["euclidean_norm"]
    df["_q1"]  = grp.transform("quantile", 0.25)
    df["_q3"]  = grp.transform("quantile", 0.75)
    df["_iqr"] = df["_q3"] - df["_q1"]
    df["lower_fence"] = df["_q1"] - iqr_multiplier * df["_iqr"]
    df["upper_fence"] = df["_q3"] + iqr_multiplier * df["_iqr"]
    df["direction"]   = np.where(
        df["euclidean_norm"] < df["lower_fence"], "LOW",
        np.where(df["euclidean_norm"] > df["upper_fence"], "HIGH", "ok")
    )
    df["is_outlier"] = df["direction"] != "ok"
    df["deviation_mm"] = np.where(
        df["direction"] == "HIGH",
        df["euclidean_norm"] - df["upper_fence"],
        np.where(df["direction"] == "LOW",
                 df["lower_fence"] - df["euclidean_norm"],
                 0.0)
    )

    outliers = df[df["is_outlier"]].copy().sort_values(
        ["session_group", "electrode", "deviation_mm"], ascending=[True, True, False]
    )

    lines = []
    W = 70

    def _hr(char="="):  lines.append(char * W)
    def _hdr(title):
        _hr()
        lines.append(f"  {title}")
        _hr()

    # ── Header ────────────────────────────────────────────────────────────────
    _hdr(f"OUTLIER REPORT — Euclidean Norm  "
         f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    lines.append(f"  Method        : Tukey IQR fence  "
                 f"(multiplier = {iqr_multiplier})")
    lines.append(f"  Fence formula : Q1 − {iqr_multiplier}×IQR  /  "
                 f"Q3 + {iqr_multiplier}×IQR")
    lines.append(f"  Total rows    : {len(df):,}")
    lines.append(f"  Outlier rows  : {len(outliers):,}  "
                 f"({100*len(outliers)/len(df):.1f} %)")
    lines.append("")

    # ── Fence table ───────────────────────────────────────────────────────────
    _hdr("FENCE VALUES PER ELECTRODE × SESSION GROUP")
    fence_tbl = (
        df.drop_duplicates(["electrode", "session_group"])
        [["session_group", "electrode", "_q1", "_q3", "_iqr",
          "lower_fence", "upper_fence"]]
        .sort_values(["session_group", "electrode"])
    )
    lines.append(f"  {'Session group':<14} {'Electrode':<12} "
                 f"{'Q1':>7} {'Q3':>7} {'IQR':>7} "
                 f"{'Lower':>8} {'Upper':>8}  n_outliers")
    lines.append("  " + "-" * (W - 2))
    for _, row in fence_tbl.iterrows():
        n_out = len(outliers[(outliers["session_group"] == row["session_group"]) &
                              (outliers["electrode"]     == row["electrode"])])
        lines.append(
            f"  {row['session_group']:<14} {row['electrode']:<12} "
            f"{row['_q1']:7.2f} {row['_q3']:7.2f} {row['_iqr']:7.2f} "
            f"{row['lower_fence']:8.2f} {row['upper_fence']:8.2f}  {n_out}"
        )
    lines.append("")

    # ── Summary by electrode ──────────────────────────────────────────────────
    _hdr("OUTLIER COUNT — BY ELECTRODE")
    tbl = outliers.groupby(["electrode", "direction"]).size().unstack(fill_value=0)
    for col in ["LOW", "HIGH"]:
        if col not in tbl.columns:
            tbl[col] = 0
    tbl["TOTAL"] = tbl["LOW"] + tbl["HIGH"]
    lines.append(f"  {'Electrode':<12} {'LOW':>6} {'HIGH':>6} {'TOTAL':>7}")
    lines.append("  " + "-" * 34)
    for idx, row in tbl.iterrows():
        lines.append(f"  {idx:<12} {row['LOW']:6d} {row['HIGH']:6d} {row['TOTAL']:7d}")
    lines.append("")

    # ── Summary by project ────────────────────────────────────────────────────
    _hdr("OUTLIER COUNT — BY PROJECT")
    proj_tbl = outliers.groupby(["project"]).size().sort_values(ascending=False)
    proj_n   = df.groupby("project").size()
    lines.append(f"  {'Project':<14} {'n_outliers':>12} {'n_total':>9} {'%':>7}")
    lines.append("  " + "-" * 46)
    for proj, n_out in proj_tbl.items():
        n_tot = proj_n.get(proj, 0)
        pct   = 100 * n_out / n_tot if n_tot else 0
        lines.append(f"  {proj:<14} {n_out:12d} {n_tot:9d} {pct:6.1f}%")
    lines.append("")

    # ── Per-subject outlier frequency ─────────────────────────────────────────
    _hdr("SUBJECTS WITH MOST OUTLIER ROWS (top 20)")
    sub_counts = (
        outliers.groupby(["subject", "project"])
        .agg(n_outliers=("euclidean_norm", "count"),
             max_dev=("deviation_mm", "max"),
             electrodes=("electrode", lambda x: ", ".join(sorted(set(x)))))
        .sort_values("n_outliers", ascending=False)
        .head(20)
        .reset_index()
    )
    lines.append(f"  {'Subject':<12} {'Project':<12} {'n_out':>6} "
                 f"{'max_dev(mm)':>12} {'Electrodes affected'}")
    lines.append("  " + "-" * (W - 2))
    for _, row in sub_counts.iterrows():
        lines.append(
            f"  {row['subject']:<12} {row['project']:<12} {row['n_outliers']:6d} "
            f"{row['max_dev']:12.2f}   {row['electrodes']}"
        )
    lines.append("")

    # ── Full detail table ─────────────────────────────────────────────────────
    _hdr("FULL OUTLIER DETAIL (sorted by session group / electrode / deviation)")
    lines.append(
        f"  {'Subject':<12} {'Project':<12} {'Session':<8} {'Run':<8} "
        f"{'Electrode':<10} {'Group':<14} {'Dir':>4} "
        f"{'Norm':>8} {'Fence':>8} {'Dev':>8}"
    )
    lines.append("  " + "-" * (W - 2))
    for _, row in outliers.iterrows():
        fence_val = (row["upper_fence"] if row["direction"] == "HIGH"
                     else row["lower_fence"])
        lines.append(
            f"  {row['subject']:<12} {row['project']:<12} "
            f"{row['session']:<8} {row['run']:<8} "
            f"{row['electrode']:<10} {row['session_group']:<14} "
            f"{row['direction']:>4} "
            f"{row['euclidean_norm']:8.2f} {fence_val:8.2f} "
            f"{row['deviation_mm']:8.2f}"
        )
    lines.append("")
    _hr()
    lines.append(f"  End of report — {len(outliers)} outlier rows across "
                 f"{outliers['subject'].nunique()} subjects")
    _hr()

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  ✔  Outlier log saved → {log_path}  "
          f"({len(outliers)} outliers in {outliers['subject'].nunique()} subjects)")

    # Clean up helper columns before returning
    outliers = outliers.drop(columns=["_q1", "_q3", "_iqr", "is_outlier"], errors="ignore")
    return outliers

def plot_by_session(df: pd.DataFrame, output_path: str,
                    n_perm: int = 10_000):
    """
    Figure 1 — Sessions 1–4 combined vs Baseline, electrode as hue.
    Permutation tests (two-sided, n_perm shuffles) are run for each
    electrode and annotated above the violins.
    """
    df = df.copy()
    df["session_group"] = df["session"].apply(
        lambda s: "Baseline"    if s == "ses-baseline" else
                  "Session 1–4" if s in ["ses-1", "ses-2", "ses-3", "ses-4"] else None
    )
    df = df.dropna(subset=["session_group", "euclidean_norm"])
    df["electrode_label"] = df["electrode"].map(ELEC_LABELS)

    elec_hue_order = [ELEC_LABELS[e] for e in ELEC_ORDER]
    n_elec = len(elec_hue_order)

    # ── Run permutation tests ─────────────────────────────────────────────────
    print(f"\n  Running permutation tests (n_perm={n_perm:,}) …")
    perm_results: dict[str, tuple[float, float]] = {}
    for raw_e, label_e in ELEC_LABELS.items():
        g_ses  = df[(df["electrode"] == raw_e) &
                    (df["session_group"] == "Session 1–4")]["euclidean_norm"].values
        g_base = df[(df["electrode"] == raw_e) &
                    (df["session_group"] == "Baseline")]["euclidean_norm"].values
        diff, p = _permutation_test(g_ses, g_base, n_perm=n_perm)
        perm_results[label_e] = (diff, p)
        sig = _sig_label(p)
        p_str = f"<{1/n_perm:.4f}" if p <= 1/n_perm else f"={p:.4f}"
        print(f"    {label_e:<12}  Δmean={diff:+.2f} mm   "
              f"p{p_str}   {sig}")

    # ── Apply Bonferroni correction ───────────────────────────────────────────
    p_threshold = 0.05 / n_elec          # Bonferroni-corrected alpha
    print(f"  Bonferroni-corrected α = {p_threshold:.4f} "
          f"(0.05 / {n_elec} electrodes)\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.violinplot(
        data         = df,
        x            = "session_group",
        y            = "euclidean_norm",
        hue          = "electrode_label",
        order        = SESSION_GROUP_ORDER,
        hue_order    = elec_hue_order,
        palette      = ELEC_PALETTE,
        inner        = "box",
        linewidth    = 0.9,
        cut          = 0,
        density_norm = "width",
        ax           = ax,
    )

    # ── Significance brackets ─────────────────────────────────────────────────
    # Each violin is centred at x ± offset depending on hue position.
    # With n_elec=4 hues seaborn spaces them symmetrically.
    ylim_data = ax.get_ylim()
    y_top     = ylim_data[1]
    y_range   = ylim_data[1] - ylim_data[0]

    # violin width fraction (seaborn default dodge)
    # We compute approximate x-centres of each hue violin
    x_ses, x_base = 0, 1          # x positions of the two groups
    half_width = 0.4               # half the full grouped-violin block width
    step = 2 * half_width / n_elec

    # Start of first violin within each group
    x_starts = {g: g_x - half_width + step / 2
                for g, g_x in zip(SESSION_GROUP_ORDER, [x_ses, x_base])}

    bracket_y_base = y_top + y_range * 0.03   # first bracket just above data
    bracket_step   = y_range * 0.07           # vertical spacing between brackets

    for ei, label_e in enumerate(elec_hue_order):
        diff, p = perm_results[label_e]
        sig     = _sig_label(p)
        color   = ELEC_PALETTE[label_e]

        # x-centres of this electrode's violin in each group
        xc_ses  = x_starts["Session 1–4"] + ei * step
        xc_base = x_starts["Baseline"]    + ei * step
        bracket_y = bracket_y_base + ei * bracket_step

        _draw_sig_bracket(ax, xc_ses, xc_base, bracket_y, sig,
                          color=color, lw=1.4, tip_len=y_range * 0.012)

        # Annotate exact p next to the bracket
        p_txt = (f"p<{1/n_perm:.4f}" if p <= 1/n_perm
                 else f"p={p:.4f}")
        ax.text(
            (xc_ses + xc_base) / 2,
            bracket_y + y_range * 0.03,
            f"Δ={diff:+.1f} mm  {p_txt}",
            ha="center", va="bottom", fontsize=7.5,
            color=color, style="italic",
        )

    # ── N annotations beneath x-ticks ────────────────────────────────────────
    counts = df.groupby("session_group")["euclidean_norm"].count()
    for i, ses in enumerate(SESSION_GROUP_ORDER):
        n = counts.get(ses, 0)
        ax.text(i, ylim_data[0] + y_range * 0.01, f"n={n}",
                ha="center", va="bottom", fontsize=9, color="dimgray")

    # ── Expand y-axis to fit brackets ─────────────────────────────────────────
    n_brackets = len(elec_hue_order)
    ax.set_ylim(ylim_data[0],
                bracket_y_base + (n_brackets - 1) * bracket_step
                + y_range * 0.18)

    ax.set_title("Euclidean Norm: Session 1\u20134 vs Baseline, by Electrode\n"
                 "(Permutation test, two-sided; brackets coloured by electrode)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Session Group", labelpad=8)
    ax.set_ylabel("Euclidean Norm (mm)", labelpad=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(SESSION_GROUP_ORDER)

    ax.legend(title="Electrode", bbox_to_anchor=(1.01, 1),
              loc="upper left", frameon=True, fontsize=10, title_fontsize=11)

    sns.despine()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Figure 1 saved → {output_path}")


# ── Figure 2 — session × project hue ──────────────────────────────────────────

def plot_by_session_and_project(df: pd.DataFrame, output_path: str):
    session_order = _session_label_order(df)

    # Only keep projects that actually appear in the data
    present_projects = sorted(df["project"].unique(),
                              key=lambda p: (p == "Sham", p))
    palette = {p: PALETTE_PROJECT.get(p, "#999999") for p in present_projects}

    fig, ax = plt.subplots(figsize=(14, 6))

    sns.violinplot(
        data=df,
        x="session_label", y="euclidean_norm",
        hue="project",
        order=session_order,
        hue_order=present_projects,
        palette=palette,
        inner="box",
        linewidth=0.9,
        cut=0,
        ax=ax,
        density_norm="width",   # same width for all violins regardless of N
    )

    ax.set_title("Euclidean Norm of Electrode Positions by Session and Project",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Session", labelpad=8)
    ax.set_ylabel("Euclidean Norm (mm)", labelpad=8)
    ax.set_xticks(range(len(session_order)))
    ax.set_xticklabels(session_order, rotation=0)

    # Legend outside plot
    handles = [
        mpatches.Patch(color=palette[p], label=p)
        for p in present_projects
    ]
    ax.legend(
        handles=handles,
        title="Project",
        bbox_to_anchor=(1.01, 1), loc="upper left",
        frameon=True, fontsize=10, title_fontsize=11,
    )

    # N per session (total, pooled across projects) beneath x-ticks
    counts = df.groupby("session_label")["euclidean_norm"].count()
    ylim = ax.get_ylim()
    for i, ses in enumerate(session_order):
        n = counts.get(ses, 0)
        ax.text(i, ylim[0] + 0.3, f"n={n}",
                ha="center", va="bottom", fontsize=8, color="dimgray")

    sns.despine(left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Figure 2 saved → {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 02_Merge_tables_and_Corret_Naming.py now tags its outputs with the
    # network, so this script must be told which one to plot. Figures and the
    # outlier report are tagged the same way: two networks plotted into
    # identically named PNGs is the kind of mix-up that is invisible until a
    # figure reaches a reviewer.
    import argparse
    SCRIPT_DIR = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(
        description="Violin plots of the Euclidean norm for ONE network.")
    ap.add_argument("--network", default="proposed",
                    help="which network's corrected table to plot; also tags "
                         "the figures and the outlier report")
    ap.add_argument("--input", default=None,
                    help="explicit corrected wide CSV, overriding --network")
    ap.add_argument("--tables", default=None,
                    help="Tables directory (default: ./Tables)")
    ap.add_argument("--output-dir", dest="output_dir", default=None,
                    help="Figures directory (default: ./Figures)")
    ap.add_argument("--list", action="store_true",
                    help="list the corrected tables available, then exit")
    args = ap.parse_args()

    tables_dir = Path(args.tables) if args.tables else SCRIPT_DIR / "Tables"
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "Figures"
    NETWORK = args.network

    if args.list:
        hits = sorted(tables_dir.glob(
            "corrected_electrode_positions_with_baseline_wide*.csv"))
        print(f"\n  Corrected tables in {tables_dir}:\n")
        for h in hits:
            print(f"    {h.name}")
        if not hits:
            print("    (none -- run 02_Merge_tables_and_Corret_Naming.py first)")
        print()
        raise SystemExit(0)

    if args.input:
        input_csv = Path(args.input)
    else:
        input_csv = (tables_dir /
                     f"corrected_electrode_positions_with_baseline_wide_{NETWORK}.csv")
        if not input_csv.exists():
            # Fall back to the untagged name produced before 02 became
            # network-aware, so older outputs still plot.
            legacy = tables_dir / "corrected_electrode_positions_with_baseline_wide.csv"
            if legacy.exists():
                print(f"  NOTE: no per-network table for '{NETWORK}'; "
                      f"using the untagged legacy file\n        {legacy.name}")
                input_csv = legacy
    if not input_csv.exists():
        raise SystemExit(
            f"Input not found: {input_csv}\n"
            f"Run: python 02_Merge_tables_and_Corret_Naming.py --network {NETWORK}\n"
            f"or use --list to see what is available.")

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{NETWORK}_{datetime.now().strftime('%Y%m%d')}"

    print(f"\n{'='*55}")
    print(f"  Euclidean Norm Violin Plots")
    print(f"{'='*55}")
    print(f"  Network: {NETWORK}")
    print(f"  Input  : {input_csv}")
    print(f"  Output : {output_dir}")
    print(f"{'='*55}\n")

    # Load
    df_raw = pd.read_csv(input_csv)
    print(f"  Loaded {len(df_raw):,} rows, {df_raw['subject'].nunique()} subjects.")

    # Prepare
    df = prepare_data(df_raw)

    # Print project summary
    proj_counts = df.groupby("project")["subject"].nunique().sort_index()
    print("\n  Subject counts per project:")
    for proj, n in proj_counts.items():
        print(f"    {proj:<12}: {n} subjects")
    print()

    # Outlier log
    log_path = output_dir / f"outlier_report_{stamp}.txt"
    write_outlier_log(df, str(log_path))

    # Figure 1 — session only
    fig1_path = output_dir / f"violinplot_euclidean_norm_by_session_{stamp}.png"
    plot_by_session(df, str(fig1_path))

    # Figure 2 — session + project hue
    fig2_path = output_dir / f"violinplot_euclidean_norm_by_session_project_{stamp}.png"
    plot_by_session_and_project(df, str(fig2_path))

    print(f"\n  Done.\n")