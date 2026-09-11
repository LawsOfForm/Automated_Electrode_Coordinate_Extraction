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
       baseline table constructed from sham-condition pickle
       files
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


def construct_baseline_coordinate_table(tables):
    root_table    = '/media/MeMoSLAP_Mesh2/PDF_Report_Generation'
    pickle_files  = glob.glob(
        os.path.join(root_table, 'sham', '02-ANALYSIS', '**', '*.pkl'),
        recursive=True
    )

    dict_data = {}
    pkl_iter  = (tqdm(pickle_files, desc="Loading baseline pickles", unit="file")
                 if TQDM_AVAILABLE else pickle_files)

    for file in pkl_iter:
        folder_str       = os.path.basename(os.path.dirname(file))
        Exp, tgt, sub_id = folder_str.split('_')[:3]
        sub              = f'sub-{sub_id}'
        with open(file, 'rb') as f:
            data = pickle.load(f)
        key = list(data[2].keys())[0]
        dict_data[f'{Exp}_{tgt}_{sub}'] = {
            'anode'  : list(data[1]),
            'cathode1': list(data[2][key][0]),
            'cathode2': list(data[2][key][1]),
            'cathode3': list(data[2][key][2]),
        }

    df_template = pd.DataFrame.from_dict(dict_data, orient='index').reset_index()
    df_template[['exp', 'condition', 'subject']] = (
        df_template['index'].str.split('_', expand=True)
    )
    df_template['session'] = 'ses-baseline'
    df_template['run']     = 'run-baseline'
    df_template.to_csv(os.path.join(tables, 'baseline_coordinate_table.csv'), index=False)
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

    # ── 2. Prepare ground-truth table (target only) ───────────────────────────
    gt_wide = df_ground_truth_wide[
        df_ground_truth_wide['condition'] == 'target'
    ].copy().reset_index(drop=True)

    # Validate: each subject should have exactly one target row
    counts = gt_wide.groupby('subject').size()
    multi  = counts[counts > 1]
    if not multi.empty:
        print(f"  ⚠  {len(multi)} subjects have >1 target baseline row "
              f"— keeping the first per subject: {multi.index.tolist()[:10]}")
        gt_wide = gt_wide.groupby('subject', group_keys=False).first().reset_index()

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

            # Ground-truth coords for this subject (single target-baseline row)
            gt_row    = gt_wide[gt_wide['subject'] == subject].iloc[0]
            gt_coords = {e: _parse_coord_value(gt_row[e]) for e in electrodes}

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

                _correct_one_row(
                    subject, ses, run, auto_long, gt_coords, gt_long,
                    electrodes, df_auto
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


def _correct_one_row(subject, session, run, auto_long, gt_coords, gt_long,
                     electrodes, df_auto):
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

        df_auto.loc[mask, 'condition'] = 'target'

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

    else:
        # No swap needed — just mark condition
        mask = df_auto['subject'] == subject
        if session is not None and 'session' in df_auto.columns:
            mask &= df_auto['session'] == session
        if run is not None and 'run' in df_auto.columns:
            mask &= df_auto['run'] == run
        df_auto.loc[mask, 'condition'] = 'target'


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
        if all(c in wide.columns for c in ['X', 'Y', 'Z']):
            wide['euclidean_norm'] = np.linalg.norm(wide[['X', 'Y', 'Z']], axis=1)
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
        if all(c in comb_wide.columns for c in ['X', 'Y', 'Z']):
            comb_wide['euclidean_norm'] = np.linalg.norm(comb_wide[['X', 'Y', 'Z']], axis=1)
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
    """Convert target-only baseline wide DataFrame to long format."""
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

if __name__ == "__main__":

    ELECTRODES = ['anode', 'cathode1', 'cathode2', 'cathode3']

    logger     = Logger(logfile_path)
    sys.stdout = logger

    print(f"Electrode Label Correction Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    try:
        # Load auto-detected electrode positions
        df_auto = pd.read_csv(
            os.path.join(tables, 'electrode_positions_MeMoSLAP_20260407.csv')
        )

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
        save_results(corrected_df, baseline_long, tables,
                     df_auto_raw=df_auto,
                     df_baseline_wide_raw=df_baseline_wide)

        print(f"\n✔  Completed. {_stats.processed} subject/run combinations processed, "
              f"{_stats.corrected} had label swaps corrected.")

    except Exception as exc:
        import traceback
        print(f"\n✗  Error: {exc}")
        traceback.print_exc()

    finally:
        print(f"\nProcess completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log saved to: {logfile_path}")
        logger.close()
        sys.stdout = logger.terminal