"""
==============================================================
  01_Extract_coordinate.py
==============================================================
  Author  : Filip Niemann
  Contact : filip.niemann@med.uni-greifswald.de

  Questions, bug reports, and feature requests are welcome —
  please reach out by e-mail.
--------------------------------------------------------------
  DESCRIPTION
  -----------
  Extracts electrode coordinates from segmentation NIfTI files
  (*_inference.nii.gz) produced by 01_Inference_all_Subjects.py.

  For each file the script:
    1. Identifies connected clusters in the binary segmentation
    2. Validates that exactly 4 electrodes are present and
       within a plausible spatial configuration
    3. Converts the voxel centre-of-mass of each cluster to
       MNI (world) coordinates using the image affine
    4. Saves all valid results to a CSV table

  A log file is written to _tmp/ (next to this script) and
  inline progress bars are printed to the terminal.
--------------------------------------------------------------
  HOW TO USE
  ----------
  Edit base_path and Table_path at the bottom of this file,
  then run:

      python 01_Extract_coordinate.py

  The output CSV and the log file location are printed when
  the script finishes.
==============================================================
"""

import os
import sys
import glob
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import center_of_mass, label
from scipy.spatial.distance import euclidean
from scipy.ndimage import measurements
from itertools import permutations

# ============================================================
#  TMP / LOG FOLDER  — created next to this script
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
TMP_DIR    = SCRIPT_DIR / '_tmp'
TMP_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = TMP_DIR / f"log_extract_coordinates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


# ============================================================
#  LOGGER
# ============================================================

class Logger:
    """Writes timestamped messages to both terminal and log file."""

    def __init__(self, log_path: Path):
        self._fh = open(log_path, 'w', buffering=1)
        self._fh.write(
            f"=== Extract Coordinates Log  "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n"
        )

    def _ts(self):
        return datetime.now().strftime('%H:%M:%S')

    def info(self, msg: str, print_also: bool = True):
        line = f"[{self._ts()}] INFO  {msg}"
        self._fh.write(line + '\n')
        if print_also:
            print(line)

    def warn(self, msg: str, print_also: bool = True):
        line = f"[{self._ts()}] WARN  {msg}"
        self._fh.write(line + '\n')
        if print_also:
            print(line)

    def error(self, msg: str, exc: Exception = None, print_also: bool = True):
        line = f"[{self._ts()}] ERROR {msg}"
        self._fh.write(line + '\n')
        if exc is not None:
            self._fh.write(traceback.format_exc() + '\n')
        if print_also:
            print(line)

    def close(self):
        self._fh.write(
            f"\n=== Finished {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
        )
        self._fh.close()


# ============================================================
#  INLINE PROGRESS BAR
# ============================================================

class ProgressBar:
    """Inline progress bar with elapsed time and ETA."""

    def __init__(self, total: int, label: str = '', width: int = 35):
        self.total   = total
        self.label   = label
        self.width   = width
        self.current = 0
        self._start  = time.time()
        self._render()

    def _render(self):
        frac   = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * frac)
        bar    = '█' * filled + '░' * (self.width - filled)
        elapsed = timedelta(seconds=int(time.time() - self._start))

        if self.current > 0:
            avg_s   = (time.time() - self._start) / self.current
            eta_str = f"ETA {timedelta(seconds=int(avg_s * (self.total - self.current)))}"
        else:
            eta_str = "ETA --:--:--"

        sys.stdout.write(
            f'\r  {self.label}  [{bar}] {self.current}/{self.total}'
            f'  elapsed {elapsed}  {eta_str}   '
        )
        sys.stdout.flush()

    def update(self, step: int = 1):
        self.current = min(self.current + step, self.total)
        self._render()

    def done(self):
        self.current = self.total
        self._render()
        sys.stdout.write('\n')
        sys.stdout.flush()


# ============================================================
#  CORE FUNCTIONS  (logic unchanged from original)
# ============================================================

def find_nifti_files(base_path):
    pattern = os.path.join(base_path, "sub-*", "unzipped", "*inference.nii.gz")
    # Method comparison study:
    # pattern = os.path.join(base_path, "sub-*", "electrode_extraction",
    #                        "ses*", "run*", "petra_inference.nii.gz")
    return glob.glob(pattern)


def voxel_to_mni(voxel_coords, affine):
    return nib.affines.apply_affine(affine, voxel_coords)


def find_electrode_clusters(img_data):
    labeled_array, num_features = label(img_data > 0)
    clusters = {}

    if num_features > 4:
        cluster_sizes = measurements.sum(
            img_data > 0, labeled_array, index=range(1, num_features + 1)
        )
        average_cluster_size = np.mean(cluster_sizes)
        size_threshold = 0.2 * average_cluster_size

        for i in range(1, num_features + 1):
            if cluster_sizes[i - 1] >= size_threshold:
                cluster_mask = labeled_array == i
                clusters[i] = {
                    'coords':         np.array(np.where(cluster_mask)).T,
                    'center_of_mass': center_of_mass(cluster_mask),
                    'size':           cluster_sizes[i - 1],
                }
    else:
        for i in range(1, num_features + 1):
            cluster_mask = labeled_array == i
            clusters[i] = {
                'coords':         np.array(np.where(cluster_mask)).T,
                'center_of_mass': center_of_mass(cluster_mask),
            }

    return clusters


def is_valid_configuration(clusters, counters, subject, session, run):
    tag = f'{subject}_{session}_{run}'

    if len(clusters) != 4:
        n = len(clusters)
        if   n == 3: key = 'three_mask_detected'
        elif n == 2: key = 'two_mask_detected'
        elif n == 1: key = 'one_mask_detected'
        elif n == 0: key = 'no_mask_detected'
        else:        key = 'more_then_four_mask_detected'
        counters[key] += 1
        counters[key + '_sub'].append(tag)
        return False, counters

    centers = [c['center_of_mass'] for c in clusters.values()]
    for perm in permutations(centers):
        distances = [euclidean(perm[0], s) for s in perm[1:]]
        if all(5 <= d <= 70 for d in distances):
            return True, counters

    return False, counters


# ============================================================
#  MAIN PROCESSING
# ============================================================

def process_nifti_files(base_path, log: Logger):

    log.info(f"Scanning for NIfTI files under: {base_path}")
    nifti_files = find_nifti_files(base_path)

    if not nifti_files:
        log.warn("No inference NIfTI files found. Check base_path and folder structure.")
        return pd.DataFrame()

    nifti_files = sorted(nifti_files)
    log.info(f"Found {len(nifti_files)} inference file(s). Starting extraction.\n")

    results  = []
    counters = {
        'total_images':                     0,
        'valid_configurations':             0,
        'invalid_configurations':           0,
        'three_mask_detected':              0,
        'two_mask_detected':                0,
        'one_mask_detected':                0,
        'no_mask_detected':                 0,
        'more_then_four_mask_detected':     0,
        'three_mask_detected_sub':          [],
        'two_mask_detected_sub':            [],
        'one_mask_detected_sub':            [],
        'no_mask_detected_sub':             [],
        'more_then_four_mask_detected_sub': [],
    }

    bar = ProgressBar(total=len(nifti_files), label='Extracting coordinates', width=35)

    for file_path in nifti_files:
        counters['total_images'] += 1

        # ── Parse subject / session / run from path ──────────
        parts   = file_path.split(os.sep)
        subject = parts[-4]   # sub-XXXX
        # session and run live inside the filename for this layout
        session_matches = [p for p in parts if p.startswith('ses-')]
        run_matches     = [p for p in parts if p.startswith('run-')]
        session = session_matches[0] if session_matches else parts[-3]
        run     = run_matches[0]     if run_matches     else parts[-2]

        try:
            # ── Load NIfTI ───────────────────────────────────
            nii_img  = nib.load(file_path)
            affine   = nii_img.affine
            img_data = nii_img.get_fdata()

            # ── Cluster detection ────────────────────────────
            clusters = find_electrode_clusters(img_data)

            # ── Validate spatial configuration ───────────────
            is_valid, counters = is_valid_configuration(
                clusters, counters, subject, session, run
            )

            if is_valid:
                counters['valid_configurations'] += 1
                cl = list(clusters.values())

                results.append({
                    'subject':      subject,
                    'session':      session,
                    'run':          run,
                    'anode_mni':    voxel_to_mni(cl[0]['center_of_mass'], affine),
                    'cathode1_mni': voxel_to_mni(cl[1]['center_of_mass'], affine),
                    'cathode2_mni': voxel_to_mni(cl[2]['center_of_mass'], affine),
                    'cathode3_mni': voxel_to_mni(cl[3]['center_of_mass'], affine),
                })

                log.info(
                    f"OK       {subject} {session} {run}  —  "
                    f"{len(clusters)} clusters, coordinates extracted.",
                    print_also=False,
                )
            else:
                counters['invalid_configurations'] += 1
                log.warn(
                    f"INVALID  {subject} {session} {run}  —  "
                    f"{len(clusters)} cluster(s) found, configuration rejected.",
                    print_also=False,
                )

        except Exception as exc:
            counters['invalid_configurations'] += 1
            log.error(
                f"EXCEPTION  {subject} {session} {run}  —  "
                f"{file_path}  —  {exc}",
                exc=exc,
                print_also=True,
            )

        bar.update()

    bar.done()

    # ── Summary ──────────────────────────────────────────────
    summary = [
        "",
        "=" * 55,
        "  SUMMARY",
        "=" * 55,
        f"  Total images processed  : {counters['total_images']}",
        f"  Valid configurations    : {counters['valid_configurations']}",
        f"  Invalid configurations  : {counters['invalid_configurations']}",
        "  ── breakdown of invalids ──────────────────────",
        f"  No mask detected        : {counters['no_mask_detected']}",
        f"  One mask detected       : {counters['one_mask_detected']}",
        f"  Two masks detected      : {counters['two_mask_detected']}",
        f"  Three masks detected    : {counters['three_mask_detected']}",
        f"  More than four masks    : {counters['more_then_four_mask_detected']}",
        "=" * 55,
    ]
    for line in summary:
        print(line)
        log._fh.write(line + '\n')

    # Log every affected subject per invalid category
    invalid_categories = [
        ('no_mask_detected_sub',             'No mask'),
        ('one_mask_detected_sub',            'One mask'),
        ('two_mask_detected_sub',            'Two masks'),
        ('three_mask_detected_sub',          'Three masks'),
        ('more_then_four_mask_detected_sub', 'More than four masks'),
    ]
    for key, label_str in invalid_categories:
        if counters[key]:
            log._fh.write(f"\n  {label_str}:\n")
            for entry in counters[key]:
                log._fh.write(f"    {entry}\n")

    return pd.DataFrame(results)


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # ── Paths — edit here ────────────────────────────────────
    # Method comparison study:
    # base_path = '/media/Data03/Thesis/Hering/derivatives/automated_electrode_extraction'

    # MeMoSLAP:
    base_path  = '/media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction'

    # Output table folder — always sits next to this script in a Tables/ subfolder.
    # No need to edit this; it moves with the script automatically.
    Table_path = SCRIPT_DIR / 'Tables'

    output_csv = os.path.join(
        Table_path,
        f"electrode_positions_MeMoSLAP_{datetime.now().strftime('%Y%m%d')}.csv"
    )

    # ── Start logger ─────────────────────────────────────────
    log = Logger(LOG_FILE)
    log.info(f"Script started")
    log.info(f"Base path  : {base_path}")
    log.info(f"Output CSV : {output_csv}")
    log.info(f"Log file   : {LOG_FILE}\n")

    print(f"\n{'='*55}")
    print(f"  Extract Electrode Coordinates")
    print(f"{'='*55}")
    print(f"  Base path  : {base_path}")
    print(f"  Output CSV : {output_csv}")
    print(f"  Log file   : {LOG_FILE}")
    print(f"{'='*55}\n")

    # ── Run ──────────────────────────────────────────────────
    df = process_nifti_files(base_path, log)

    # ── Save results ─────────────────────────────────────────
    if not df.empty:
        os.makedirs(Table_path, exist_ok=True)
        df.to_csv(output_csv, index=False)
        msg = f"Results saved → {output_csv}  ({len(df)} rows)"
        print(f"\n  {msg}")
        log.info(msg)
    else:
        msg = "No valid results to save — CSV not written."
        print(f"\n  {msg}")
        log.warn(msg)

    print(f"  Log file   : {LOG_FILE}\n")
    log.close()