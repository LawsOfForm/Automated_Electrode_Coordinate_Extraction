"""
==============================================================
  02_Merge_tables_2.py
==============================================================
  Author  : Filip Niemann
  Contact : filip.niemann@med.uni-greifswald.de
  Questions, bug reports, and feature requests are welcome —
  please reach out by e-mail.
--------------------------------------------------------------
  DESCRIPTION
  -----------
  Merges and corrects electrode coordinate tables produced by
  01_Extract_coordinate.py.  For each subject / session / run
  combination the script:
    1. Loads automated electrode positions and a ground-truth
       baseline table constructed from the SimNIBS result
       pickles found under BOTH the sham and active folders
    2. Verifies that the detected anode is spatially surrounded
       by the three cathodes (circular configuration check)
    3. Selects the best-matching ground-truth condition
       (target vs. control) via total Euclidean distance
    4. Applies the Hungarian (Munkres) optimal assignment
       algorithm to correct any electrode-label swaps
    5. Saves corrected tables in both long and wide CSV format,
       including an Euclidean norm column
  A log file (coordinate_correction.log) is written next to
  this script.  Inline progress bars and a live statistics
  panel are printed to the terminal during processing.
--------------------------------------------------------------
  HOW TO USE
  ----------
  Edit the path variables in the __main__ block at the bottom,
  then run:
      python 02_Merge_tables_2.py
  Output CSVs and the log file location are printed when the
  script finishes.
==============================================================
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os
import pickle
import glob
import re
import copy
import sys
import time
from datetime import datetime

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# Progress / display helpers -------------------------------------------------
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Minimal fallback so the rest of the code never branches
    class tqdm:                                         # noqa: F811
        def __init__(self, iterable=None, **kw):
            self._it = iterable or []
            self.total = kw.get('total', None)
            self.desc = kw.get('desc', '')
            self.n = 0
        def __iter__(self):
            for item in self._it:
                self.n += 1
                yield item
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): self.n += n
        def set_postfix(self, **kw): pass
        def set_description(self, s): self.desc = s

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
    )
    from rich.layout import Layout
    from rich import box
    RICH_AVAILABLE = True
    _console = Console()
except ImportError:
    RICH_AVAILABLE = False
    _console = None

# ---------------------------------------------------------------------------
# Logging — mirrors output to a file without silencing the terminal
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ---------------------------------------------------------------------------
# Runtime statistics collector (shared mutable state for the live panel)
# ---------------------------------------------------------------------------
class RunStats:
    """Accumulates per-subject statistics for the live display."""
    def __init__(self):
        self.processed   = 0
        self.corrected   = 0
        self.warnings    = 0
        self.skipped     = 0
        self.anode_ok    = 0
        self.anode_wrong = 0
        self.star_ok     = 0     # post-correction star check passed
        self.star_revert = 0     # swap reverted because it broke the star
        self.mirror_warn = 0     # baseline may be contralateral
        self.cond_switched = 0   # control montage fitted better than target
        self.cond_ambiguous = 0  # the two montages fitted almost equally well
        # How each subject's montage was chosen. Kept on the stats object
        # because it is written inside the correction loop and read by the
        # summary, which is a different function.
        self.side_tally = {}
        self.distances   = []           # optimised distances collected so far
        self._start      = time.time()

    def elapsed(self) -> str:
        s = int(time.time() - self._start)
        return f"{s // 60:02d}:{s % 60:02d}"

    def mean_distance(self) -> str:
        if not self.distances:
            return "—"
        return f"{np.mean(self.distances):.2f} mm"

    def rich_table(self) -> "Table":
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column("Metric", style="bold cyan")
        t.add_column("Value",  style="white")
        t.add_row("Processed",          str(self.processed))
        t.add_row("Label swaps fixed",  str(self.corrected))
        t.add_row("Anode ✓ / ✗",
                  f"[green]{self.anode_ok}[/green] / [red]{self.anode_wrong}[/red]")
        t.add_row("Star ✓ / reverted",
                  f"[green]{self.star_ok}[/green] / [red]{self.star_revert}[/red]")
        t.add_row("Mirror warnings",  f"[yellow]{self.mirror_warn}[/yellow]")
        t.add_row("Condition switched", f"[cyan]{self.cond_switched}[/cyan]")
        t.add_row("Condition ambiguous", f"[yellow]{self.cond_ambiguous}[/yellow]")
        t.add_row("Warnings",           f"[yellow]{self.warnings}[/yellow]")
        t.add_row("Skipped",            f"[dim]{self.skipped}[/dim]")
        t.add_row("Mean opt. distance", self.mean_distance())
        t.add_row("Elapsed",            self.elapsed())
        return t


_stats = RunStats()

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
script_directory = Path(__file__).parent.resolve()
root   = script_directory.parent.parent.resolve()
tables = os.path.join(script_directory, 'Tables')

logfile_path = os.path.join(script_directory, 'coordinate_correction.log')

# _tmp/ lives next to this script — created on import so every function can
# use it safely without checking existence first.
TMP_DIR = script_directory / '_tmp'
TMP_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: plain progress bar that degrades gracefully when rich is absent
# ---------------------------------------------------------------------------
def _make_rich_progress(desc: str, total: int):
    """Return a rich Progress context manager or None."""
    if not RICH_AVAILABLE:
        return None
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[extra]}"),
        console=_console,
        transient=False,
    )


# ---------------------------------------------------------------------------
# Core science functions (unchanged logic, added progress hooks)
# ---------------------------------------------------------------------------

def euclidean_distance(coord1, coord2):
    return np.sqrt(np.sum((np.array(coord1) - np.array(coord2)) ** 2))


def verify_anode_configuration(electrode_coords, tolerance_factor=1.5):
    """
    Verify that the anode is surrounded by 3 cathodes in a circular fashion.

    Returns
    -------
    is_valid      : bool
    anode_idx     : int | None   index into electrode_coords.keys()
    confidence    : float        0–1
    """
    electrodes = list(electrode_coords.keys())

    valid_electrodes, valid_coords = [], []
    for electrode in electrodes:
        coords_data = electrode_coords[electrode]
        coords = (coords_data[electrode]
                  if isinstance(coords_data, dict) and electrode in coords_data
                  else coords_data)
        if (isinstance(coords, (list, np.ndarray)) and len(coords) >= 3
                and all(c is not None and not np.isnan(c) for c in coords[:3])):
            valid_electrodes.append(electrode)
            valid_coords.append(coords)

    if len(valid_electrodes) < 4:
        print(f"  ⚠  Only {len(valid_electrodes)} valid electrodes — need 4 for verification")
        return False, None, 0.0

    coords_array = np.array(valid_coords)
    try:
        dist_matrix = cdist(coords_array, coords_array)
    except Exception as exc:
        print(f"  ✗  Distance matrix error: {exc}")
        return False, None, 0.0

    best_anode_idx, best_score = None, -1
    for i in range(len(valid_electrodes)):
        anode_dists  = [dist_matrix[i, j] for j in range(len(valid_electrodes)) if j != i]
        cathode_dists = sorted(anode_dists)[:3]
        mean_cd       = np.mean(cathode_dists)
        std_cd        = np.std(cathode_dists)

        uniformity = max(0.0, min(1.0, 1 - std_cd / mean_cd)) if mean_cd > 0 else 0.0

        other_idx    = [j for j in range(len(valid_electrodes)) if j != i]
        cathode_idx  = [other_idx[k] for k in np.argsort(anode_dists)[:3]]
        pair_dists   = [dist_matrix[a, b]
                        for idx1, a in enumerate(cathode_idx)
                        for b in cathode_idx[idx1 + 1:]]

        ratio = min(np.mean(pair_dists) / mean_cd, 3) if pair_dists and mean_cd > 0 else 0.0
        score = uniformity * 0.7 + (ratio / 3) * 0.3

        if score > best_score:
            best_score, best_anode_idx = score, i

    is_valid = best_score > 0.3
    if best_anode_idx is not None:
        name = valid_electrodes[best_anode_idx]
        original_idx = electrodes.index(name) if name in electrodes else None
    else:
        original_idx = None

    return is_valid, original_idx, best_score


def construct_baseline_coordinate_table(tables, stim_folders=('sham', 'active'),
                                        on_conflict='first'):
    """Build the intended-montage table from the SimNIBS result pickles.

    Searches every folder in `stim_folders` under

        <root>/<stim>/02-ANALYSIS/**/*.pkl

    The previous version searched 'sham' only, so subjects whose optimisation
    was stored under 'active' had no baseline and were silently dropped from
    every downstream comparison.

    CONFLICTS
    ---------
    The dictionary key is Exp_condition_subject, which does NOT include the
    stimulation folder, so the same key can now arrive from both 'sham' and
    'active'. Overwriting one with the other silently -- which is what a plain
    dict assignment does -- would make the baseline depend on directory
    iteration order. Instead:

      * identical coordinates (within 0.01 mm): keep one, no message. This is
        the expected case for a montage that is physically the same and only
        differs in the current waveform;
      * differing coordinates: report the key, the distance, and both sources,
        then resolve by `on_conflict`:
            'first'  keep the folder listed earliest in stim_folders
            'skip'   drop the key entirely, so no correction is attempted
                     against an ambiguous target

    The returned table gains a `stim` column recording which folder each
    baseline came from, so a conflict can be traced afterwards.
    """
    root_table = '/media/MeMoSLAP_Mesh2/PDF_Report_Generation'

    pickle_files = []
    per_folder = {}
    for stim in stim_folders:
        hits = glob.glob(os.path.join(root_table, stim, '02-ANALYSIS',
                                      '**', '*.pkl'), recursive=True)
        per_folder[stim] = len(hits)
        pickle_files += [(stim, h) for h in hits]

    print("\n  Baseline pickles found:")
    for stim, n in per_folder.items():
        print(f"    {stim:<8} {n:>5}")
    if not pickle_files:
        print(f"    none under {root_table}/<{'|'.join(stim_folders)}>/02-ANALYSIS")
        print("    Check the path and the folder names.")

    dict_data, source, conflicts, skipped = {}, {}, [], []
    pkl_iter = (tqdm(pickle_files, desc="Loading baseline pickles", unit="file")
                if TQDM_AVAILABLE else pickle_files)

    for stim, file in pkl_iter:
        folder_str = os.path.basename(os.path.dirname(file))
        parts = folder_str.split('_')
        if len(parts) < 3:
            skipped.append((file, f"folder name '{folder_str}' is not "
                                  f"Exp_condition_subject"))
            continue
        Exp, tgt, sub_id = parts[0], parts[1], parts[2]
        sub = f'sub-{sub_id}'
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            key_inner = list(data[2].keys())[0]
            coords = {
                'anode':    list(data[1]),
                'cathode1': list(data[2][key_inner][0]),
                'cathode2': list(data[2][key_inner][1]),
                'cathode3': list(data[2][key_inner][2]),
            }
        except Exception as exc:                                # noqa: BLE001
            skipped.append((file, f"{type(exc).__name__}: {exc}"))
            continue

        key = f'{Exp}_{tgt}_{sub}'
        if key in dict_data:
            prev = dict_data[key]
            d = max(float(np.linalg.norm(np.array(prev[e]) - np.array(coords[e])))
                    for e in ('anode', 'cathode1', 'cathode2', 'cathode3'))
            if d <= 0.01:
                continue                       # same montage in both folders
            conflicts.append((key, source[key], stim, d))
            if on_conflict == 'skip':
                dict_data.pop(key, None)
                source.pop(key, None)
                continue
            continue                           # 'first': keep what we have
        dict_data[key] = coords
        source[key] = stim

    if conflicts:
        print(f"\n  CONFLICT: {len(conflicts)} subject(s) have a baseline in "
              f"more than one stimulation folder with DIFFERENT coordinates.")
        print(f"  Resolution: {'kept the first folder listed' if on_conflict=='first' else 'dropped entirely'}.")
        for key, first, second, d in conflicts[:20]:
            print(f"    {key:<28} {first} vs {second}   max difference {d:7.1f} mm")
        if len(conflicts) > 20:
            print(f"    ... and {len(conflicts)-20} more")
        print("  These are worth resolving at source: the correction step will")
        print("  match against whichever montage was kept.")

    if skipped:
        print(f"\n  {len(skipped)} pickle(s) could not be read:")
        for f, why in skipped[:10]:
            print(f"    {os.path.basename(os.path.dirname(f))}: {why}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped)-10} more")

    if not dict_data:
        raise SystemExit(
            "\n  No baseline coordinates could be loaded. Every downstream "
            "correction depends on them, so stopping here rather than "
            "producing an empty table.")

    df_template = pd.DataFrame.from_dict(dict_data, orient='index').reset_index()
    df_template[['exp', 'condition', 'subject']] = (
        df_template['index'].str.split('_', expand=True)
    )
    df_template['stim'] = df_template['index'].map(source)
    df_template['session'] = 'ses-baseline'
    df_template['run'] = 'run-baseline'
    df_template.to_csv(os.path.join(tables, 'baseline_coordinate_table.csv'),
                       index=False)

    print(f"\n  Baseline table: {len(df_template)} subject/condition entries")
    for stim, n in df_template['stim'].value_counts().items():
        print(f"    from {stim:<8} {n:>5}")
    print(f"    unique subjects {df_template['subject'].nunique()}\n")
    return df_template


def parse_coordinates(coord_str):
    # Matches integers, decimals, and scientific notation (e.g. 1.23e+02).
    # The previous regex r'-?\d+\.\d+' silently dropped integer-valued
    # coordinates (no decimal point) and mangled scientific notation by
    # stripping the exponent and reading the exponent digits as a separate
    # number (e.g. '2.007e+01' → 2.007 instead of 20.07).
    numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', coord_str)
    return [float(n) for n in numbers]


def wide_to_long(df, electrodes):
    df = df.melt(
        id_vars   =[c for c in df.columns if c not in electrodes],
        value_vars =electrodes,
        var_name  ='electrode',
        value_name='coordinates',
    )
    try:
        df['coordinates'] = df['coordinates'].apply(parse_coordinates)
    except Exception:
        print('  ⚠  Could not parse coordinates (continuing).')

    df_coordinates = df.copy()
    dims           = ['X', 'Y', 'Z']
    df[dims]       = df['coordinates'].apply(pd.Series)
    df             = df.drop(columns=['coordinates'])
    df             = df.melt(
        id_vars   =[c for c in df.columns if c not in dims],
        value_vars =dims,
        var_name  ='dimension',
        value_name='coordinates',
    )
    return df, df_coordinates


def is_dataframe_in_long_format(df,
        required_columns=('subject', 'session', 'run', 'electrode', 'dimension', 'coordinates')):
    return all(c in df.columns for c in required_columns)


def prepare_dataframe_for_processing(df, electrodes=('anode', 'cathode1', 'cathode2', 'cathode3')):
    if is_dataframe_in_long_format(df):
        print("  ℹ  DataFrame already in long format — using as-is.")
        df_coords = df.copy()
        if df_coords['coordinates'].dtype == 'object':
            try:
                df_coords['coordinates'] = df_coords['coordinates'].apply(parse_coordinates)
            except Exception:
                pass
        return df, df_coords
    print("  ℹ  DataFrame in wide format — converting to long.")
    return wide_to_long(df, electrodes)


def create_coords_df(df_2, electrode):
    df_copy = copy.deepcopy(df_2)
    if 'dimension' in df_copy.columns and not df_copy[df_copy['dimension'].isin(['X','Y','Z'])].empty:
        try:
            coords = [
                df_copy[df_copy['dimension'] == d]['coordinates'].values[0]
                if len(df_copy[df_copy['dimension'] == d]) > 0 else None
                for d in ('X', 'Y', 'Z')
            ]
        except Exception:
            coords = [None, None, None]
    else:
        try:
            arr = df_copy['coordinates'].values[0]
            coords = list(arr[:3]) if isinstance(arr, (list, np.ndarray)) and len(arr) >= 3 else [None]*3
        except Exception:
            coords = [None, None, None]
    return {electrode: coords}


def extract_coordinates(coord_dict, electrode):
    if electrode in coord_dict:
        d = coord_dict[electrode]
        return d[electrode] if isinstance(d, dict) and electrode in d else d
    return [None, None, None]


# ---------------------------------------------------------------------------
# Coordinate correction pipeline
# ---------------------------------------------------------------------------

def check_and_correct_coordinates(df_input, tables, df_ground_truth_wide,
                                   electrodes=('anode', 'cathode1', 'cathode2', 'cathode3')):
    """
    Main entry point.

    Logic
    -----
    For every subject that exists in BOTH df_input and the baseline table:
      - Take all auto-detected rows for that subject (one per segmentation run)
      - Take only the TARGET-condition baseline row for that subject as ground truth
        (control rows are never used)
      - For each auto row: apply Hungarian matching to find the optimal
        electrode-label assignment, then write corrected labels back
    The loop is strictly subject × run — no Cartesian product with other subjects.
    """
    electrodes = list(electrodes)

    # ── 1. Prepare auto-detection table ──────────────────────────────────────
    # 01_Extract_coordinate.py now writes correct subject/session/run columns
    # directly from the filename — no column-shift fix needed.
    df_auto, _ = prepare_dataframe_for_processing(df_input, electrodes)

    # ── 2. Prepare ground-truth table ────────────────────────────────────────
    # Default: the target montage only. Control montages are loaded and
    # available, but selecting between them by best fit was tested and
    # degraded the result badly (see MATCH_CONDITION). Set MATCH_CONDITION to
    # 'best' to re-enable it.
    gt_wide = df_ground_truth_wide.copy().reset_index(drop=True)

    if MATCH_CONDITION == 'hemisphere':
        n = gt_wide.groupby('subject')['condition'].nunique()
        print(f"  Matching by hemisphere. {int((n > 1).sum())} subject(s) have "
              f"more than one candidate montage; for these the side of the "
              f"extracted electrodes decides.")
    elif MATCH_CONDITION == 'any':
        # One baseline per subject is the expected case. More than one means a
        # subject appears in both arms, which the study design excludes, so it
        # is reported rather than resolved silently.
        per_sub = gt_wide.groupby('subject')['condition'].nunique()
        multi = per_sub[per_sub > 1]
        if len(multi):
            print(f"  ⚠  {len(multi)} subject(s) have baselines for more than "
                  f"one condition, which the study design does not expect: "
                  f"{', '.join(multi.index[:8])}"
                  f"{' ...' if len(multi) > 8 else ''}")
            print("     Their first baseline will be used; verify these before "
                  "trusting their deviations.")
        counts = gt_wide.groupby('condition')['subject'].nunique()
        print("  Baseline conditions in use: " +
              ", ".join(f"{c} {n}" for c, n in counts.items()))
    elif MATCH_CONDITION == 'target':
        n_before = gt_wide['subject'].nunique()
        gt_wide = gt_wide[gt_wide['condition'] == 'target']
        n_after = gt_wide['subject'].nunique()
        if n_before != n_after:
            print(f"  Matching against the TARGET montage only; "
                  f"{n_before - n_after} subject(s) have no target baseline "
                  f"and will be skipped.")

    n_cond = gt_wide.groupby('subject')['condition'].nunique()
    multi_cond = n_cond[n_cond > 1]
    if len(multi_cond):
        print(f"  {len(multi_cond)} subject(s) have more than one baseline "
              f"condition; the montage that fits the extraction will be used.")
    for subj, grp in gt_wide.groupby('subject'):
        dup = grp['condition'].duplicated()
        if dup.any():
            print(f"  ⚠  {subj}: {dup.sum()} duplicate row(s) for the same "
                  f"condition — keeping the first of each.")
    gt_wide = gt_wide.drop_duplicates(['subject', 'condition'], keep='first')

    gt_long, _ = wide_to_long(gt_wide, electrodes)

    # ── 3. Find shared subjects ───────────────────────────────────────────────
    auto_subjects = set(df_auto['subject'].unique())
    gt_subjects   = set(gt_wide['subject'].unique())
    shared        = sorted(auto_subjects & gt_subjects)
    only_auto     = sorted(auto_subjects - gt_subjects)

    print(f"  ℹ  {len(shared)} subjects with auto-detections AND target baseline.")
    if only_auto:
        print(f"  ℹ  {len(only_auto)} subjects skipped (no target baseline): "
              f"{only_auto[:10]}{'…' if len(only_auto)>10 else ''}")

    # ── 4. Mark condition column ──────────────────────────────────────────────
    df_auto['condition'] = 'unknown'

    # ── 5. Loop — strictly subject × (session, run) ───────────────────────────
    # Each unique subject/session/run triplet from the auto-detection table is
    # matched independently against that subject's single target baseline.
    # No cross-subject comparisons occur.
    if RICH_AVAILABLE:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=38),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("[dim]{task.fields[extra]}"),
            console=_console,
            transient=False,
        )
        task = progress.add_task(
            "Correcting electrode labels …",
            total=len(shared),
            extra="",
        )
        live_ctx = Live(_build_live_layout(progress), console=_console, refresh_per_second=4)
    else:
        progress = live_ctx = None

    def _advance(subj):
        if progress:
            progress.update(task, extra=subj)
            progress.advance(task)

    with (live_ctx if live_ctx else _null_ctx()):
        for subject in shared:
            _advance(subject)

            # Every baseline montage available for this subject, keyed by
            # condition. Which one applies is decided per image below.
            gt_rows = gt_wide[gt_wide['subject'] == subject]
            gt_candidates = {
                r['condition']: {e: _parse_coord_value(r[e]) for e in electrodes}
                for _, r in gt_rows.iterrows()
            }
            # The label written to the output is the condition this subject was
            # actually assigned, so a downstream reader can tell target from
            # control rather than seeing everything labelled 'target'.
            subject_condition = (list(gt_candidates)[0]
                                 if len(gt_candidates) == 1 else 'ambiguous')
            if not gt_candidates:
                continue

            # All auto-detected rows for this subject
            sub_auto = df_auto[df_auto['subject'] == subject]

            # Iterate over every (session, run) combination for this subject
            combo_cols = [c for c in ['session', 'run'] if c in sub_auto.columns]
            combos = sub_auto[combo_cols].drop_duplicates().values.tolist()

            for combo in combos:
                ses = combo[combo_cols.index('session')] if 'session' in combo_cols else None
                run = combo[combo_cols.index('run')]     if 'run'     in combo_cols else None

                mask = pd.Series([True] * len(sub_auto), index=sub_auto.index)
                if ses is not None:
                    mask &= sub_auto['session'] == ses
                if run is not None:
                    mask &= sub_auto['run'] == run
                auto_long = sub_auto[mask]

                # Choose the montage this image actually fits, then correct
                # the labels against it.
                auto_flat_probe = _flatten_auto(auto_long, electrodes)
                if MATCH_CONDITION == 'hemisphere' and auto_flat_probe:
                    cond_used, how, why = choose_condition_by_side(
                        subject, auto_flat_probe, gt_candidates, electrodes)
                    gt_coords = gt_candidates[cond_used]
                    cond_info = dict(chosen=cond_used, how=how, why=why,
                                     residuals={}, margin=float('inf'),
                                     ambiguous=(how == "default"),
                                     subject_condition=cond_used)
                    _stats.side_tally[how] = _stats.side_tally.get(how, 0) + 1
                    if why:
                        print(f"  ?  {subject} | {ses} | {run}: {why}; "
                              f"defaulting to '{cond_used}'")
                elif (MATCH_CONDITION != 'best' or auto_flat_probe is None
                        or len(gt_candidates) == 1):
                    cond_used = next(iter(gt_candidates))
                    gt_coords = gt_candidates[cond_used]
                    cond_info = dict(chosen=cond_used,
                                     residuals={cond_used: float('nan')},
                                     margin=float('inf'), ambiguous=False,
                                     subject_condition=subject_condition)
                else:
                    cond_used, gt_coords, cond_info = _pick_baseline_condition(
                        auto_flat_probe, gt_candidates, electrodes,
                        subject, ses, run)

                _correct_one_row(
                    subject, ses, run, auto_long, gt_coords, gt_long,
                    electrodes, df_auto, cond_info=cond_info
                )
                _stats.processed += 1

            if live_ctx:
                live_ctx.update(_build_live_layout(progress))

    return df_auto, gt_long


def _parse_coord_value(val):
    """Parse a coordinate value that may be a list, ndarray, or string."""
    if isinstance(val, (list, np.ndarray)):
        return [float(x) for x in val]
    numbers = re.findall(r'-?\d+\.?\d*(?:e[+-]?\d+)?', str(val))
    return [float(x) for x in numbers]


# A control montage must beat the target one by at least this margin, in mean
# millimetres per electrode, before the choice is treated as informative.
# Below it the two montages fit the extraction about equally well and the
# selection is effectively arbitrary.
CONDITION_MARGIN_MM = 10.0

# Which baseline condition to match against.
#   'target'  use only the target montage (the original behaviour)
#   'best'    also consider the control montage and keep whichever fits
#
# 'target' is the default because 'best' was tested and made the result
# markedly worse. Enabling it raised the number of matched electrodes from
# 9,556 to 15,792, and every project's median deviation rose with it -- P6 from
# 12.1 mm to 100.7 mm. The added matches were overwhelmingly spurious: the
# deviation distributions became bimodal with a second mode at 120-180 mm,
# which is the signature of an electrode matched against a montage on the
# other side of the head. Selecting the montage that minimises the residual
# cannot detect that, because a wrong montage still has a best-fitting
# assignment. Mismatches are therefore FLAGGED rather than resolved; see the
# `note` column.
MATCH_CONDITION = 'hemisphere'
#
# Every subject has BOTH a target and a control pickle: SimNIBS optimised both
# montages for everyone, and which one was delivered is not recorded in them.
# A choice is therefore unavoidable. The two montages sit on OPPOSITE SIDES of
# the head, so the side the extracted electrodes are on identifies which was
# applied.
#
# This is not the same as choosing by best fit, and the difference matters. Fit
# is the quantity the choice is later used to explain, so selecting on it is
# circular and cannot detect its own failure; that version was tested and made
# the result markedly worse (matched electrodes 9,556 -> 15,792, P6's median
# deviation 12.1 -> 100.7 mm). Side is a categorical fact about the image,
# independent of how well anything fits, and a montage on the wrong hemisphere
# cannot be argued into agreement.
#
# Validated against an independent record of the stimulation arm for 77
# subjects of project P3: 589 of 589 images and 77 of 77 subjects classified
# correctly, every session of every subject agreeing, and the nearest image
# still 35 mm from the decision boundary. Control montages sat at a mean
# left-right coordinate of +48.7 mm and target montages at -61.6 mm, with no
# overlap.
#
# Alternatives, none recommended:
#   'any'    use whichever single baseline exists (wrong here: there are two)
#   'target' always the target montage; mislabels the entire control arm
#   'best'   choose by fit. Do not use, for the reason above.

# The two montages must be separated by at least this much along the left-right
# axis, and the extracted montage must lie at least this far from the midline,
# before the side is treated as informative. Below either threshold the subject
# is left on the target montage and noted, rather than guessed at.
HEMI_SEPARATION_MIN_MM = 30.0
HEMI_SIDE_MIN_MM = 15.0


def _mean_lr(coords, electrodes):
    """Mean left-right coordinate of a montage."""
    return float(np.mean([np.asarray(coords[e], dtype=float)[0]
                          for e in electrodes]))


def choose_condition_by_side(subject, auto_flat, candidates, electrodes):
    """Which of the candidate montages is on the same side as the extraction.

    Returns (condition, how, note); `how` records the criterion so the output
    states on what basis each subject was matched.
    """
    if len(candidates) == 1:
        return next(iter(candidates)), "only baseline", ""

    ex = _mean_lr(auto_flat, electrodes)
    sides = {c: _mean_lr(v, electrodes) for c, v in candidates.items()}
    sep = max(sides.values()) - min(sides.values())
    fallback = "target" if "target" in candidates else next(iter(candidates))

    if sep < HEMI_SEPARATION_MIN_MM:
        return fallback, "default", (
            f"montages only {sep:.0f} mm apart; side is uninformative")
    if abs(ex) < HEMI_SIDE_MIN_MM:
        return fallback, "default", (
            f"extraction {abs(ex):.0f} mm from the midline; side unclear")
    same = [c for c, x in sides.items() if np.sign(x) == np.sign(ex)]
    if len(same) != 1:
        return fallback, "default", "side ambiguous"
    return same[0], "hemisphere", ""
#
# 'any' is correct for this study: each subject was assigned to ONE stimulation
# arm, so there is exactly one baseline per subject and nothing to choose
# between. Restricting to 'target', as an earlier version did, silently dropped
# every subject in the control arm or matched them against a montage that was
# never applied to them -- which is where the large rigid offsets in some
# projects came from.
#
# 'target' reproduces that earlier behaviour. 'best' additionally picks between
# several candidate montages by fit; it is retained only for completeness and
# should not be used. It was tested and made the result markedly worse (matched
# electrodes 9,556 -> 15,792, every project's median deviation rose, P6 from
# 12.1 mm to 100.7 mm), because a wrong montage still has a best-fitting
# assignment and the criterion cannot detect its own failure.

# Mean residual above which a row is noted as suspect rather than silently
# used. It does not change the matching, only the note.
NOTE_RESIDUAL_MM = 40.0


def _flatten_auto(auto_long, electrodes):
    """{electrode: [x, y, z]} for one image, or None if it is incomplete.

    Uses the same create_coords_df / extract_coordinates path as
    _correct_one_row, so the coordinates compared here are exactly the ones
    the correction will act on. Re-parsing the long table independently would
    risk the two disagreeing.
    """
    auto_coords = {}
    for e in electrodes:
        auto_coords[e] = create_coords_df(
            auto_long[auto_long['electrode'] == e], e)
    out = {}
    for e in electrodes:
        c = extract_coordinates(auto_coords, e)
        if c is None or None in c or any(np.isnan(x) for x in c):
            return None
        out[e] = c
    return out


def _assignment_residual(auto_flat, gt_coords, electrodes):
    """Mean per-electrode distance under the OPTIMAL label assignment.

    The optimal assignment is used rather than the identity so that a montage
    is not penalised merely because its labels are permuted -- which is the
    very thing the correction step exists to fix. What is being compared here
    is the geometry, not the labelling.
    """
    n = len(electrodes)
    D = np.zeros((n, n))
    for i, ge in enumerate(electrodes):
        for j, ae in enumerate(electrodes):
            D[i, j] = np.linalg.norm(np.asarray(gt_coords[ge], dtype=float) -
                                     np.asarray(auto_flat[ae], dtype=float))
    ri, ci = linear_sum_assignment(D)
    return float(D[ri, ci].sum() / n)


def _pick_baseline_condition(auto_flat, candidates, electrodes,
                             subject, session, run):
    """Choose between the target and control montages for one image.

    A subject can have both a P<n>_target_<id> and a P<n>_control_<id> pickle.
    Which montage a given session actually received is not recorded in the
    image, and matching against the wrong one produces a large residual and,
    worse, a label assignment that is optimal for a montage that was never
    applied.

    This picks the montage the extracted electrodes actually fit.

    A CAUTION THAT BELONGS IN THE METHODS SECTION: selecting the baseline that
    minimises the residual and then reporting that residual as accuracy is
    circular, and will bias the reported deviation downwards. It is defensible
    only when the two montages are far enough apart that the choice is
    unambiguous, which is why the margin between them is recorded for every
    image and small margins are flagged rather than silently resolved. If the
    two montages sit close together, prefer the condition recorded in the study
    metadata over the one chosen here.

    Returns (condition, coords, info).
    """
    res = {c: _assignment_residual(auto_flat, co, electrodes)
           for c, co in candidates.items()}
    best = min(res, key=res.get)
    others = [v for k, v in res.items() if k != best]
    margin = (min(others) - res[best]) if others else float('inf')

    info = dict(chosen=best, residuals=res, margin=margin,
                ambiguous=bool(others) and margin < CONDITION_MARGIN_MM)
    if len(candidates) > 1:
        detail = ", ".join(f"{c} {v:.1f} mm" for c, v in sorted(res.items()))
        if info["ambiguous"]:
            print(f"  ⚠  {subject} | {session} | {run}: target and control "
                  f"montages fit almost equally well ({detail}; margin "
                  f"{margin:.1f} mm < {CONDITION_MARGIN_MM:.0f} mm). Using "
                  f"'{best}', but the choice is not informative — prefer the "
                  f"recorded condition if one exists.")
            _stats.cond_ambiguous += 1
        elif best != 'target':
            print(f"  ↪  {subject} | {session} | {run}: '{best}' montage fits "
                  f"better than 'target' ({detail}). Using it, and labelling "
                  f"the row 'target' as the reference condition.")
            _stats.cond_switched += 1
    return best, candidates[best], info


def _star_ok(coords_by_label):
    """Is the electrode labelled 'anode' the central one?

    Independent of the Hungarian matching: the central electrode of a 3x1
    montage is the one whose three spokes subtend the largest minimum angle.
    Curvature of the scalp changes those angles but not which point maximises
    them, so no absolute threshold is involved.

    Returns (ok, central_label, min_angle_deg).
    """
    names = list(coords_by_label)
    if len(names) != 4:
        return False, None, float("nan")
    pts = {k: np.asarray(v, dtype=float) for k, v in coords_by_label.items()}
    best = None
    for h in names:
        sp = [pts[o] - pts[h] for o in names if o != h]
        if min(np.linalg.norm(s) for s in sp) < 1e-6:
            continue
        ang = []
        for i in range(3):
            for j in range(i + 1, 3):
                ca = float(np.dot(sp[i], sp[j]) /
                           (np.linalg.norm(sp[i]) * np.linalg.norm(sp[j]) + 1e-9))
                ang.append(float(np.degrees(np.arccos(np.clip(ca, -1, 1)))))
        if best is None or min(ang) > best[1]:
            best = (h, min(ang))
    if best is None:
        return False, None, float("nan")
    return best[0] == 'anode', best[0], best[1]


# Mean distance to the baseline above which the baseline is reported as
# possibly belonging to a different, or contralateral, montage. A correct
# match is a few millimetres to a few tens; a mirrored 70 mm montage lands
# near 50 mm and still passes the 300 mm Hungarian gate, so the gate alone
# cannot catch it.
MIRROR_WARN_MM = 40.0


# Thresholds for the shape diagnosis below, in millimetres. Calibrated by
# simulation: with 3 mm of placement noise on four electrodes, the rigid-fit
# RMSD and the mean pairwise-distance difference both sit near 4-5 mm at the
# 95th percentile, so 8 mm separates noise from a real geometric difference
# without being tight enough to fire on ordinary placement variation.
SHAPE_TOL_MM = 8.0
SPAN_TOL_MM = 8.0


def _rigid_diagnosis(auto_flat, gt_coords, electrodes):
    """Separate a wrong SPACE from a wrong MONTAGE.

    A coregistration or reference-image mismatch moves the whole montage
    rigidly: it translates and rotates every electrode together and leaves the
    six inter-electrode distances untouched. A genuinely different montage
    changes those distances, and no rigid transform can absorb it. Fitting the
    optimal rotation and translation (Kabsch) and then asking how much residual
    is left therefore distinguishes the two, which the raw residual alone
    cannot.

    ONE DEGENERACY, worth knowing: the montage is planar, and reflecting a
    planar object in its own plane equals rotating it 180 degrees out of plane.
    A mirrored montage therefore also looks rigid here. Shape cannot separate
    the two; only the anatomy can, which is what the opposite-side-of-midline
    test does. A large fitted rotation combined with a large translation is the
    usual signature.

    Returns a dict, or None if the configuration is incomplete.
    """
    if len(auto_flat) != 4 or any(e not in gt_coords for e in electrodes):
        return None
    A = np.array([auto_flat[e] for e in electrodes], dtype=float)
    B = np.array([gt_coords[e] for e in electrodes], dtype=float)

    Ac, Bc = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(Ac.T @ Bc)
    dsign = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, dsign]) @ U.T
    rigid_rmsd = float(np.sqrt(((Ac @ R.T - Bc) ** 2).sum(1).mean()))
    rot_deg = float(np.degrees(np.arccos(
        np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))))
    trans = B.mean(0) - A.mean(0)

    pa = sorted(float(np.linalg.norm(A[i] - A[j]))
                for i in range(4) for j in range(i + 1, 4))
    pb = sorted(float(np.linalg.norm(B[i] - B[j]))
                for i in range(4) for j in range(i + 1, 4))
    shape_diff = float(np.mean([abs(x - y) for x, y in zip(pa, pb)]))
    span_diff = float(abs(np.mean(pa) - np.mean(pb)))

    raw = float(np.linalg.norm(A - B, axis=1).mean())
    rigid_like = (shape_diff < SHAPE_TOL_MM and rigid_rmsd < SHAPE_TOL_MM
                  and span_diff < SPAN_TOL_MM)
    return dict(raw=raw, rigid_rmsd=rigid_rmsd, shape_diff=shape_diff,
                span_diff=span_diff, span_extracted=float(np.mean(pa)),
                span_baseline=float(np.mean(pb)), rot_deg=rot_deg,
                trans=trans, trans_norm=float(np.linalg.norm(trans)),
                rigid_like=rigid_like)


def _mirror_warning(auto_flat, gt_coords):
    """Report a baseline that looks like it belongs to the other hemisphere.

    The Hungarian assignment is optimal for whatever target it is given; it
    cannot tell that the target is on the wrong side of the head. Two signals
    are combined, neither conclusive alone: the mean residual after matching,
    and whether the anode and its baseline differ in the sign of their first
    coordinate by more than a plausible placement error.
    """
    common = [e for e in auto_flat if e in gt_coords]
    if len(common) < 4:
        return None
    resid = float(np.mean([
        np.linalg.norm(np.asarray(auto_flat[e], dtype=float) -
                       np.asarray(gt_coords[e], dtype=float))
        for e in common]))
    ax = float(auto_flat['anode'][0])
    bx = float(gt_coords['anode'][0])
    opposite = (ax * bx < 0) and (abs(ax) > 12) and (abs(bx) > 12)
    if resid > MIRROR_WARN_MM or opposite:
        return dict(residual=resid, opposite=opposite)
    return None


def _correct_one_row(subject, session, run, auto_long, gt_coords, gt_long,
                     electrodes, df_auto, cond_info=None):
    """
    Apply Hungarian label correction to one subject/session/run block.

    auto_long : long-format rows for this subject/session/run
    gt_coords : {electrode: [x,y,z]}  target baseline for this subject
    """
    # Build auto coordinate dict from long-format rows
    auto_coords = {}
    for electrode in electrodes:
        e_rows = auto_long[auto_long['electrode'] == electrode]
        coords = create_coords_df(e_rows, electrode)
        auto_coords[electrode] = coords

    # Check we have all 4 electrodes
    auto_flat = {}
    for e in electrodes:
        c = extract_coordinates(auto_coords, e)
        if c is None or None in c or any(np.isnan(x) for x in c):
            print(f"  ⚠  {subject} | {session} | {run}: missing coords for {e} — skipping.")
            _stats.warnings += 1
            return
        auto_flat[e] = c

    # ── Anode configuration check ─────────────────────────────────────────
    is_valid, detected_anode_idx, confidence = verify_anode_configuration(auto_flat)
    elec_list = list(auto_flat.keys())
    if detected_anode_idx is not None and detected_anode_idx < len(elec_list):
        detected = elec_list[detected_anode_idx]
        ok = detected == 'anode'
        print(f"  {subject} | {session} | {run}  anode={detected} "
              f"conf={confidence:.2f} {'✓' if ok else '✗'}")
        if ok: _stats.anode_ok    += 1
        else:  _stats.anode_wrong += 1

    # Record which baseline montage was used, and how decisively, so the
    # choice is auditable in the output rather than only in the console.
    # ── Data-quality note ─────────────────────────────────────────────────
    # A single column that names why a row is suspect, so downstream analyses
    # can exclude it without re-deriving the reasoning. Empty means nothing was
    # detected. Multiple reasons are joined with '+' so the column stays one
    # value per row and can be filtered with a substring test.
    _notes = []

    ci = cond_info or {}
    _cond_cols = dict(
        baseline_condition_used=ci.get('chosen', 'target'),
        stimulation_arm=ci.get('subject_condition', ''),
        baseline_margin_mm=(round(ci['margin'], 2)
                            if ci.get('margin') not in (None, float('inf'))
                            else ''),
        baseline_ambiguous='yes' if ci.get('ambiguous') else 'no',
        note='')
    for _c, _v in _cond_cols.items():
        if _c not in df_auto.columns:
            df_auto[_c] = ''
    for _cname, _res in (ci.get('residuals') or {}).items():
        col = f'baseline_residual_{_cname}_mm'
        if col not in df_auto.columns:
            df_auto[col] = ''
    if ci.get('ambiguous'):
        _notes.append('ambiguous_condition')

    # ── Contralateral / wrong-montage baseline warning ────────────────────
    # ── Rigid-vs-montage diagnosis ────────────────────────────────────────
    rd = _rigid_diagnosis(auto_flat, gt_coords, electrodes)
    if rd:
        for _c, _v in (('rigid_rmsd_mm', round(rd['rigid_rmsd'], 2)),
                       ('shape_diff_mm', round(rd['shape_diff'], 2)),
                       ('span_extracted_mm', round(rd['span_extracted'], 2)),
                       ('span_baseline_mm', round(rd['span_baseline'], 2)),
                       ('fit_rotation_deg', round(rd['rot_deg'], 1)),
                       ('fit_translation_mm', round(rd['trans_norm'], 2))):
            _cond_cols[_c] = _v
            if _c not in df_auto.columns:
                df_auto[_c] = ''
        if rd['raw'] > NOTE_RESIDUAL_MM:
            if rd['rigid_like']:
                # Shape intact: the montage is right, its frame is not.
                _notes.append('rigid_offset?')
                print(f"  ↔  {subject} | {session} | {run}: residual "
                      f"{rd['raw']:.0f} mm but the montage SHAPE is intact "
                      f"(rigid fit {rd['rigid_rmsd']:.1f} mm, shape diff "
                      f"{rd['shape_diff']:.1f} mm). Consistent with a "
                      f"coordinate-space or reference-image mismatch rather "
                      f"than a misplaced montage; recoverable by a transform.")
            else:
                _notes.append('different_montage?')
                print(f"  ✗  {subject} | {session} | {run}: residual "
                      f"{rd['raw']:.0f} mm and the geometry differs "
                      f"(shape diff {rd['shape_diff']:.1f} mm, spans "
                      f"{rd['span_extracted']:.0f} vs "
                      f"{rd['span_baseline']:.0f} mm). The baseline appears to "
                      f"describe a different montage.")

    mw = _mirror_warning(auto_flat, gt_coords)
    if mw:
        bits = []
        if mw['opposite']:
            bits.append("anode on the OPPOSITE side of the midline from its baseline")
            _notes.append('contralateral_baseline?')
        if mw['residual'] > NOTE_RESIDUAL_MM:
            _notes.append('high_residual')
        _cond_cols['baseline_residual_mm'] = round(mw['residual'], 2)
        if 'baseline_residual_mm' not in df_auto.columns:
            df_auto['baseline_residual_mm'] = ''
        bits.append(f"mean residual {mw['residual']:.0f} mm")
        print(f"  ⚠  {subject} | {session} | {run}: possible contralateral or "
              f"mismatched baseline — {'; '.join(bits)}. The label assignment "
              f"below is optimal for THIS baseline; verify it visually.")
        _stats.mirror_warn += 1

    # ── Hungarian matching ────────────────────────────────────────────────
    n = len(electrodes)
    D = np.zeros((n, n))
    for i, ge in enumerate(electrodes):
        for j, ae in enumerate(electrodes):
            D[i, j] = np.linalg.norm(np.array(gt_coords[ge]) - np.array(auto_flat[ae]))

    row_ind, col_ind = linear_sum_assignment(D)
    mapping = {}          # correct_label -> current_auto_label
    total_dist = 0.0
    for i, j in zip(row_ind, col_ind):
        total_dist += D[i, j]
        if electrodes[i] != electrodes[j]:
            mapping[electrodes[i]] = electrodes[j]

    _stats.distances.append(total_dist)

    # ── Distance sanity gate ──────────────────────────────────────────────
    # If the optimal assignment still produces a very large total distance
    # the baseline entry is for a completely different montage or hemisphere
    # and no correction should be applied.  The condition stays 'unknown'.
    MAX_HUNGARIAN_DIST_MM = 300.0   # 4 electrodes × avg 75 mm — clearly wrong
    if total_dist > MAX_HUNGARIAN_DIST_MM:
        print(f"  ⚠  {subject} | {session} | {run}: "
              f"Hungarian Σd={total_dist:.1f} mm > {MAX_HUNGARIAN_DIST_MM:.0f} mm threshold — "
              f"baseline montage mismatch, correction skipped (condition stays 'unknown').")
        _stats.warnings += 1
        return

    # Only apply if Hungarian is better than identity
    identity_dist = sum(D[i, i] for i in range(n))
    if total_dist < identity_dist and mapping:
        print(f"    Swaps: {mapping}  Σd {total_dist:.1f} < identity {identity_dist:.1f}")
        _stats.corrected += 1

        # Build mask using all available identifiers
        mask = df_auto['subject'] == subject
        if session is not None and 'session' in df_auto.columns:
            mask &= df_auto['session'] == session
        if run is not None and 'run' in df_auto.columns:
            mask &= df_auto['run'] == run

        # The condition written here is the montage actually selected for this
        # subject, carried in cond_info by the caller. Hardcoding 'target'
        # mislabels the entire control arm, and any downstream join on
        # condition then matches those subjects to a montage never applied to
        # them. `subject_condition` is a local of the caller and is not in
        # scope here, which is why it is read from cond_info instead.
        _cond_label = (ci.get('subject_condition') or ci.get('chosen')
                       or 'target')
        if _cond_label == 'ambiguous':
            _cond_label = 'target'
        df_auto.loc[mask, 'condition'] = _cond_label
        _cond_cols['note'] = '+'.join(_notes)
        for _c, _v in _cond_cols.items():
            df_auto.loc[mask, _c] = _v
        for _cname, _res in (ci.get('residuals') or {}).items():
            df_auto.loc[mask, f'baseline_residual_{_cname}_mm'] = round(_res, 2)

        # ── Atomic label rename (two-phase to handle cyclic swaps) ────────
        # mapping = {correct_label: src_label}, so renaming means:
        #   the rows currently carrying src_label should become correct_label.
        # A direct single-pass rename breaks cyclic permutations (e.g.
        # anode↔cathode1): halfway through, the source rows have already been
        # overwritten and the second rename finds nothing.
        # Fix: first rename all affected rows to unique temp labels, then
        # rename from temp to the final labels.  No coordinate rewriting is
        # needed — the coordinates belong to the rows and travel with them.
        _TMP = '__swap__'
        remap = {src: correct for correct, src in mapping.items()}

        # Phase 1 — park each row that needs relabeling under a temp label
        for src_label in remap:
            elec_mask = mask & (df_auto['electrode'] == src_label)
            if not df_auto[elec_mask].empty:
                df_auto.loc[elec_mask, 'electrode'] = src_label + _TMP

        # Phase 2 — assign the final correct labels
        for src_label, correct_label in remap.items():
            elec_mask = mask & (df_auto['electrode'] == src_label + _TMP)
            if not df_auto[elec_mask].empty:
                df_auto.loc[elec_mask, 'electrode'] = correct_label

        # ── Post-swap star verification ──────────────────────────────────
        # verify_anode_configuration ran BEFORE the matching, on the labels as
        # extracted. Nothing checked the labels the matching produced, so a
        # swap could leave the electrode called 'anode' off-centre and the run
        # would still be recorded as a success. Re-check here, against the
        # montage geometry rather than against the baseline, so the two tests
        # are independent. If the swap breaks the star it is undone: a
        # geometrically impossible labelling is worse than the original, and
        # leaving it in place would put it into the coordinate tables.
        swapped_coords = {corr: auto_flat[src] for corr, src in mapping.items()}
        for e in electrodes:
            swapped_coords.setdefault(e, auto_flat[e])
        ok_star, central, min_ang = _star_ok(swapped_coords)
        if ok_star:
            _stats.star_ok += 1
        else:
            print(f"  ⚠  {subject} | {session} | {run}: swap REVERTED — after "
                  f"relabelling, '{central}' is the central electrode, not the "
                  f"anode (min angle {min_ang:.0f}°). Original labels kept.")
            _stats.star_revert += 1
            _stats.corrected -= 1
            _notes.append('star_failed_swap_reverted')
            # Undo phase 2 then phase 1, in reverse, restoring the original
            # labels exactly.
            for src_label, correct_label in remap.items():
                em = mask & (df_auto['electrode'] == correct_label)
                if not df_auto[em].empty:
                    df_auto.loc[em, 'electrode'] = src_label + _TMP
            for src_label in remap:
                em = mask & (df_auto['electrode'] == src_label + _TMP)
                if not df_auto[em].empty:
                    df_auto.loc[em, 'electrode'] = src_label

    else:
        # No swap applied. Still record whether the labels as extracted form a
        # valid star, so the statistic covers every row rather than only the
        # corrected ones.
        ok_star, central, min_ang = _star_ok(auto_flat)
        if ok_star:
            _stats.star_ok += 1
        else:
            print(f"  ⚠  {subject} | {session} | {run}: no swap applied, but "
                  f"'{central}' is the central electrode, not the anode "
                  f"(min angle {min_ang:.0f}°).")
            _stats.anode_wrong += 1
            _notes.append('anode_not_central')

        mask = df_auto['subject'] == subject
        if session is not None and 'session' in df_auto.columns:
            mask &= df_auto['session'] == session
        if run is not None and 'run' in df_auto.columns:
            mask &= df_auto['run'] == run
        # The condition written here is the montage actually selected for this
        # subject, carried in cond_info by the caller. Hardcoding 'target'
        # mislabels the entire control arm, and any downstream join on
        # condition then matches those subjects to a montage never applied to
        # them. `subject_condition` is a local of the caller and is not in
        # scope here, which is why it is read from cond_info instead.
        _cond_label = (ci.get('subject_condition') or ci.get('chosen')
                       or 'target')
        if _cond_label == 'ambiguous':
            _cond_label = 'target'
        df_auto.loc[mask, 'condition'] = _cond_label
        _cond_cols['note'] = '+'.join(_notes)
        for _c, _v in _cond_cols.items():
            df_auto.loc[mask, _c] = _v
        for _cname, _res in (ci.get('residuals') or {}).items():
            df_auto.loc[mask, f'baseline_residual_{_cname}_mm'] = round(_res, 2)


def _build_live_layout(progress) -> "Panel":
    from rich.columns import Columns
    content = Columns([progress, _stats.rich_table()], equal=False, expand=True)
    return Panel(content, title="[bold]Electrode Label Correction — Live Status[/bold]",
                 border_style="blue")


from contextlib import contextmanager

@contextmanager
def _null_ctx():
    yield



# ---------------------------------------------------------------------------
# Merge audit log
# ---------------------------------------------------------------------------

def write_merge_audit_log(
    df_auto_raw:       pd.DataFrame,
    df_baseline_wide:  pd.DataFrame,
    combined:          pd.DataFrame,
    tmp_dir:           Path = TMP_DIR,
) -> Path:
    """
    Write a structured audit log to ``tmp_dir`` summarising the merge
    between the auto-detection table and the baseline table.

    Sections
    --------
    1. Subject counts — how many unique subjects are in each input table
       and in the merged result, with the overlap and symmetric difference.
    2. Subjects missing a baseline session — every subject present in the
       merged table that has no row with ``session == 'ses-baseline'``.
       These subjects will have no ground-truth anchor for comparison.
    3. Subjects in baseline only — present in the baseline table but absent
       from the auto-detection table (no coordinates were extracted for them).

    Parameters
    ----------
    df_auto_raw      : original auto-detection DataFrame *before* correction
                       (wide or long, must contain a ``subject`` column)
    df_baseline_wide : baseline DataFrame as loaded / constructed
                       (must contain ``subject`` and optionally ``session``)
    combined         : the merged long-format DataFrame produced by
                       ``save_results`` (corrected auto + baseline long)
    tmp_dir          : directory to write the log into (created if absent)

    Returns
    -------
    log_path : Path to the written log file
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = tmp_dir / f'merge_audit_{timestamp}.txt'

    # ── Collect subject sets ─────────────────────────────────────────────────
    auto_subs     = set(df_auto_raw['subject'].dropna().unique())
    baseline_subs = set(df_baseline_wide['subject'].dropna().unique())
    combined_subs = set(combined['subject'].dropna().unique())

    both          = sorted(auto_subs & baseline_subs)
    only_auto     = sorted(auto_subs - baseline_subs)
    only_baseline = sorted(baseline_subs - auto_subs)

    # ── Subjects with no ses-baseline row in the merged table ────────────────
    # A subject "has a baseline" when at least one row in `combined` carries
    # session == 'ses-baseline'.  Any subject lacking such a row has no
    # ground-truth anchor after the merge.
    if 'session' in combined.columns:
        has_baseline = set(
            combined.loc[combined['session'] == 'ses-baseline', 'subject']
            .dropna().unique()
        )
    else:
        # Fall back to condition column if session is absent
        has_baseline = set(
            combined.loc[combined.get('condition', pd.Series(dtype=str)) == 'target',
                         'subject']
            .dropna().unique()
        ) if 'condition' in combined.columns else set()

    no_baseline_in_merged = sorted(combined_subs - has_baseline)

    # ── Write log ────────────────────────────────────────────────────────────
    sep  = '=' * 62
    sep2 = '-' * 62

    lines = [
        sep,
        '  MERGE AUDIT LOG',
        f'  Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        sep,
        '',
        '  SECTION 1 — Subject counts',
        sep2,
        f'  Auto-detection table       : {len(auto_subs):>5} subjects',
        f'  Baseline table             : {len(baseline_subs):>5} subjects',
        f'  ── after merge ────────────────────────────────────',
        f'  Combined (unique subjects) : {len(combined_subs):>5} subjects',
        f'  In BOTH tables             : {len(both):>5} subjects',
        f'  Auto-only (no baseline)    : {len(only_auto):>5} subjects',
        f'  Baseline-only (no auto)    : {len(only_baseline):>5} subjects',
        '',
    ]

    # Section 2 — subjects missing ses-baseline after the merge
    lines += [
        f'  SECTION 2 — Subjects with NO baseline session after merge',
        sep2,
        f'  Count : {len(no_baseline_in_merged)}',
        '',
    ]
    if no_baseline_in_merged:
        lines.append('  These subjects exist in the merged table but have no')
        lines.append('  row with session == "ses-baseline":')
        lines.append('')
        for sub in no_baseline_in_merged:
            lines.append(f'    {sub}')
    else:
        lines.append('  ✓  All subjects in the merged table have a baseline session.')
    lines.append('')

    # Section 3 — subjects only in the auto table (never matched to baseline)
    lines += [
        f'  SECTION 3 — Auto-only subjects (no baseline row available)',
        sep2,
        f'  Count : {len(only_auto)}',
        '',
    ]
    if only_auto:
        lines.append('  Coordinates were extracted for these subjects but no')
        lines.append('  matching baseline entry was found — Hungarian correction')
        lines.append('  could not be applied:')
        lines.append('')
        for sub in only_auto:
            lines.append(f'    {sub}')
    else:
        lines.append('  ✓  Every auto-detected subject has a baseline entry.')
    lines.append('')

    # Section 4 — subjects only in baseline (no coords extracted)
    lines += [
        f'  SECTION 4 — Baseline-only subjects (no auto-detection)',
        sep2,
        f'  Count : {len(only_baseline)}',
        '',
    ]
    if only_baseline:
        lines.append('  These subjects are in the baseline table but no electrode')
        lines.append('  coordinates were extracted for them:')
        lines.append('')
        for sub in only_baseline:
            lines.append(f'    {sub}')
    else:
        lines.append('  ✓  All baseline subjects also have auto-detected coordinates.')
    lines += ['', sep, '']

    with open(log_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))

    return log_path


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def _add_xyz_column(wide, decimals=3):
    """Add a combined [x, y, z] column and order the coordinate columns.

    The wide table already carries X, Y and Z separately, which is what an
    analysis wants. A person reading a single electrode wants the triple in
    one place instead of recombining three columns by eye, so both forms are
    provided.

    FORMAT: a JSON list, "[74.127, -62.028, 5.331]". A CSV cell is always
    text, so what matters is whether that text parses without custom code.
    This form is both valid JSON and a valid Python literal, so it reads back
    with json.loads or ast.literal_eval and goes straight into numpy:

        import json, numpy as np, pandas as pd
        df = pd.read_csv(path)
        xyz = np.array(df["coordinates_xyz"].apply(json.loads).tolist())

    A parenthesised "(x, y, z)" would be a Python tuple but not JSON, and the
    numpy-repr style "[74.127 -62.028 5.331]" written by 01 is neither, which
    is why 02 needs a regex to parse it. Square brackets with commas avoid
    that entirely.

    The column is named coordinates_xyz rather than 'coordinates', because the
    LONG table uses 'coordinates' for a single scalar value; reusing the name
    for a triple would make the two files disagree about what it means.
    """
    if not all(c in wide.columns for c in ("X", "Y", "Z")):
        return wide
    xyz = wide[["X", "Y", "Z"]].astype(float).round(decimals)
    wide["coordinates_xyz"] = [
        "" if any(pd.isna(v) for v in row) else
        f"[{row[0]:.{decimals}f}, {row[1]:.{decimals}f}, {row[2]:.{decimals}f}]"
        for row in xyz.to_numpy()
    ]
    wide["euclidean_norm"] = np.linalg.norm(
        wide[["X", "Y", "Z"]].astype(float), axis=1)
    # Put the coordinate columns together, in the order a reader expects.
    lead = [c for c in wide.columns
            if c not in ("X", "Y", "Z", "coordinates_xyz", "euclidean_norm",
                         "note")]
    tail = ["X", "Y", "Z", "coordinates_xyz", "euclidean_norm"]
    if "note" in wide.columns:
        tail.append("note")          # last, so it reads as an annotation
    return wide[lead + tail]


def save_results(corrected_df, baseline_long, tables, suffix="",
                 df_auto_raw=None, df_baseline_wide_raw=None):
    """
    Save corrected results.
    corrected_df        : long-format DataFrame returned by check_and_correct_coordinates
    baseline_long       : long-format baseline (target only) returned by check_and_correct_coordinates
    df_auto_raw         : original (uncorrected) auto-detection DataFrame — used for
                          the merge audit log (optional; audit is skipped if None)
    df_baseline_wide_raw: original baseline wide DataFrame — used for the merge
                          audit log (optional; audit is skipped if None)
    """
    print("\n  Saving corrected data …")
    corrected_df = corrected_df.drop_duplicates()

    # ── Long format (no baseline) ─────────────────────────────────────────────
    out_long = os.path.join(tables, f'corrected_electrode_positions_no_baseline_long{suffix}.csv')
    corrected_df.to_csv(out_long, index=False)
    print(f"    {out_long}")

    # ── Wide format (no baseline) ─────────────────────────────────────────────
    pivot_cols = [c for c in
                  ['subject', 'session', 'run', 'method', 'electrode', 'condition']
                  if c in corrected_df.columns]
    dedup_cols = pivot_cols + ['dimension']
    try:
        wide = (corrected_df
                .drop_duplicates(subset=dedup_cols, keep='first')
                .pivot(index=pivot_cols, columns='dimension', values='coordinates')
                .reset_index())
        wide = _add_xyz_column(wide)
        out_wide = os.path.join(tables, f'corrected_electrode_positions_no_baseline_wide{suffix}.csv')
        wide.to_csv(out_wide, index=False)
        print(f"    {out_wide}")
    except Exception as e:
        print(f"  ⚠  Wide format failed: {e}")

    # ── Combined with baseline ────────────────────────────────────────────────
    combined = pd.concat([corrected_df, baseline_long], axis=0, ignore_index=True)
    dedup_key = [c for c in ['subject', 'session', 'run', 'electrode', 'condition', 'dimension']
                 if c in combined.columns]
    combined = combined.drop_duplicates(subset=dedup_key, keep='first')

    out_comb_long = os.path.join(tables, f'corrected_electrode_positions_with_baseline_long{suffix}.csv')
    combined.to_csv(out_comb_long, index=False)
    print(f"    {out_comb_long}")

    pivot_cols_full = [c for c in
                       ['subject', 'session', 'run', 'method', 'electrode', 'condition']
                       if c in combined.columns]
    try:
        comb_wide = (combined
                     .drop_duplicates(subset=pivot_cols_full + ['dimension'], keep='first')
                     .pivot(index=pivot_cols_full, columns='dimension', values='coordinates')
                     .reset_index())
        comb_wide = _add_xyz_column(comb_wide)
        out_comb_wide = os.path.join(tables, f'corrected_electrode_positions_with_baseline_wide{suffix}.csv')
        comb_wide.to_csv(out_comb_wide, index=False)
        print(f"    {out_comb_wide}")
    except Exception as e:
        print(f"  ⚠  Combined wide format failed: {e}")

    # ── Merge audit log ───────────────────────────────────────────────────────
    if df_auto_raw is not None and df_baseline_wide_raw is not None:
        try:
            audit_path = write_merge_audit_log(
                df_auto_raw, df_baseline_wide_raw, combined
            )
            print(f"    Merge audit log → {audit_path}")
        except Exception as e:
            print(f"  ⚠  Could not write merge audit log: {e}")

    print("  ✔ All files saved.")


def _baseline_wide_to_long(bl_wide, electrodes):
    """Convert the baseline wide DataFrame to long format.

    Keeps `stim` (sham / active) and `exp` so that 04 and 06 can filter on
    them; they are properties of the baseline, not of the extraction, and are
    otherwise lost at this step."""
    rows = []
    for _, row in bl_wide.iterrows():
        for e in electrodes:
            coords = row.get(e)
            if not isinstance(coords, (list, np.ndarray)):
                try:
                    coords = _parse_coord_value(coords)
                except Exception:
                    continue
            for dim, val in zip(['X', 'Y', 'Z'], coords):
                rows.append({'subject': row.get('subject'),
                             'session': row.get('session'),
                             'run':     row.get('run'),
                             'condition': row.get('condition', 'target'),
                             # Carried through so downstream analyses can be
                             # restricted to one stimulation folder without
                             # re-reading baseline_coordinate_table.csv.
                             'stim':      row.get('stim', ''),
                             'exp':       row.get('exp', ''),
                             'electrode': e,
                             'dimension': dim,
                             'coordinates': float(val)})
    return pd.DataFrame(rows)


def _remove_duplicates(df):
    print("  Checking for duplicates …")
    df = df.drop_duplicates()
    pivot_columns = [c for c in
                     ['subject', 'session', 'run', 'electrode', 'condition', 'dimension']
                     if c in df.columns]
    idx_dups = df[df.duplicated(subset=pivot_columns, keep=False)]
    if len(idx_dups):
        print(f"  ⚠  {len(idx_dups)} pivot-index duplicates — keeping first.")
        df = df.drop_duplicates(subset=pivot_columns, keep='first')
    print(f"  ✔  {len(df)} rows after deduplication.")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Input resolution
#
# 01_Extract_coordinate_ablation_final.py writes one table per network,
#     Tables/electrode_positions_<network>_<YYYYMMDD>.csv
# so this script must be told WHICH network to correct, and must tag its own
# outputs the same way. Without that, running a second network silently
# overwrites the first network's corrected tables and every downstream figure
# becomes unattributable.
# ---------------------------------------------------------------------------

KNOWN_NETWORKS = ['proposed', 'b_no_attention', 'a_baseline',
                  'd_increased', 'c_reduced']


def find_input_table(tables_dir, network, explicit=None):
    """Newest electrode_positions_<network>_*.csv, or an explicit path."""
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"Input table not found: {explicit}")
        return explicit
    hits = sorted(glob.glob(os.path.join(
        tables_dir, f'electrode_positions_{network}_*.csv')))
    if not hits:
        raise SystemExit(
            f"No table for network '{network}' in {tables_dir}\n"
            f"Expected: electrode_positions_{network}_<YYYYMMDD>.csv\n"
            f"Run 01_Extract_coordinate_ablation_final.py --network {network} first, "
            f"or pass --input.")
    if len(hits) > 1:
        print(f"  NOTE: {len(hits)} tables for '{network}'; using the newest.")
        for h in hits[:-1]:
            print(f"        ignoring {os.path.basename(h)}")
    return hits[-1]


def list_available(tables_dir):
    print(f"\n  Tables in {tables_dir}:\n")
    found = False
    for net in KNOWN_NETWORKS:
        hits = sorted(glob.glob(os.path.join(
            tables_dir, f'electrode_positions_{net}_*.csv')))
        for h in hits:
            n = sum(1 for _ in open(h)) - 1
            print(f"    {net:<16} {os.path.basename(h):<48} {n:>6} rows")
            found = True
    if not found:
        print("    (none -- run 01_Extract_coordinate_ablation_final.py first)")
    print()


if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="Correct electrode labels for ONE network's coordinate "
                    "table and write network-tagged outputs.")
    ap.add_argument("--network", default="proposed",
                    help="which network's table to correct; also tags every "
                         "output file so networks cannot overwrite each other")
    ap.add_argument("--input", default=None,
                    help="explicit input CSV, overriding the --network lookup")
    ap.add_argument("--tables", default=None,
                    help="Tables directory (default: ./Tables beside this script)")
    ap.add_argument("--list", action="store_true",
                    help="list the per-network tables available, then exit")
    args = ap.parse_args()

    if args.tables:
        tables = os.path.expanduser(args.tables)
    if args.list:
        list_available(tables)
        raise SystemExit(0)

    NETWORK = args.network
    SUFFIX = f"_{NETWORK}"
    input_csv = find_input_table(tables, NETWORK, args.input)
    logfile_path = os.path.join(script_directory,
                                f'coordinate_correction_{NETWORK}.log')


    ELECTRODES = ['anode', 'cathode1', 'cathode2', 'cathode3']

    logger     = Logger(logfile_path)
    sys.stdout = logger

    print(f"Electrode Label Correction Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    try:
        # Load auto-detected electrode positions
        print(f"  Network : {NETWORK}")
        print(f"  Input   : {input_csv}")
        df_auto = pd.read_csv(input_csv)
        print(f"  Rows    : {len(df_auto):,}\n")

        # Load baseline (built from pickles if available, otherwise from CSV)
        try:
            df_baseline_wide = construct_baseline_coordinate_table(tables)
        except Exception:
            print("  ℹ  Could not build baseline from pickles — loading CSV directly.")
            df_baseline_wide = pd.read_csv(
                os.path.join(tables, 'baseline_coordinate_table.csv')
            )
            for e in ELECTRODES:
                df_baseline_wide[e] = df_baseline_wide[e].apply(_parse_coord_value)

        print("Starting electrode label correction …")
        corrected_df, baseline_long = check_and_correct_coordinates(
            df_auto, tables, df_baseline_wide, ELECTRODES
        )
        # The network tag travels with the data as well as the filename, so a
        # merged or re-opened table is still self-describing.
        if 'network' not in corrected_df.columns:
            corrected_df.insert(0, 'network', NETWORK)
        save_results(corrected_df, baseline_long, tables, suffix=SUFFIX,
                     df_auto_raw=df_auto,
                     df_baseline_wide_raw=df_baseline_wide)

        print(f"\n✔  Completed. {_stats.processed} subject/run combinations processed, "
              f"{_stats.corrected} had label swaps corrected.")
        print(f"   Star check: {_stats.star_ok} passed, "
              f"{_stats.star_revert} swap(s) reverted because the swap broke "
              f"the montage geometry.")
        for _lbl, _key in (("rigid_offset?", 'rigid_offset?'),
                           ("different_montage?", 'different_montage?')):
            if 'note' in df_auto.columns:
                _n = int(df_auto['note'].astype(str).str.contains(
                    _key, regex=False).sum())
                if _n:
                    print(f"   {_n} row(s) noted '{_lbl}'.")
        if _stats.side_tally:
            print("   Condition chosen by: " + ", ".join(
                f"{k} {v}" for k, v in sorted(_stats.side_tally.items())))
            if _stats.side_tally.get("default"):
                print(f"   {_stats.side_tally['default']} image(s) could not be "
                      f"resolved by side and kept the target montage; these "
                      f"carry a note.")
        n_noted = 0
        if 'note' in df_auto.columns:
            n_noted = int((df_auto['note'].astype(str).str.len() > 0).sum())
        if n_noted:
            print(f"   {n_noted} row(s) carry a data-quality note. The `note` "
                  f"column names the reason; downstream analyses can exclude "
                  f"them with, for example, --exclude-note "
                  f"contralateral_baseline? high_residual.")
        if _stats.cond_switched:
            print(f"   {_stats.cond_switched} image(s) fitted the CONTROL montage "
                  f"better than the target and were matched against it. The "
                  f"chosen montage is recorded per row in "
                  f"'baseline_condition_used'.")
        if _stats.cond_ambiguous:
            print(f"   {_stats.cond_ambiguous} image(s) where the two montages "
                  f"fitted almost equally well — the selection there is not "
                  f"informative; prefer the recorded study condition.")
        if _stats.cond_switched or _stats.cond_ambiguous:
            print("   NOTE: selecting the baseline that minimises the residual "
                  "and then reporting that residual as accuracy is circular. "
                  "Report the deviation against the RECORDED condition where "
                  "one exists, and use this selection only to identify images "
                  "whose recorded condition looks wrong.")
        if _stats.mirror_warn:
            print(f"   {_stats.mirror_warn} row(s) flagged as a possible "
                  f"contralateral or mismatched baseline — review these before "
                  f"using their coordinates.")

    except Exception as exc:
        import traceback
        print(f"\n✗  Error: {exc}")
        traceback.print_exc()

    finally:
        print(f"\nProcess completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log saved to: {logfile_path}")
        logger.close()
        sys.stdout = logger.terminal