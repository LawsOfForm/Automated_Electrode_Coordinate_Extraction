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
       corregistered native space (world) coordinates using the image affine
    4. Saves all valid results to a CSV table

  If more than 4 clusters are found after size-based filtering,
  the script applies a cascade of recovery strategies before
  marking a configuration invalid:

  • n = 5, fast path — `filter_fifth_cluster` tries to identify
    and drop a single spurious noise cluster.

  • n = 5–8, general recovery — `recover_four_electrodes` searches
    exhaustively for the best partition of n clusters into exactly
    4 electrode groups, covering every combination of:
      – dropping 1–4 spurious noise blobs,
      – merging 1–3 pairs of nearby fragments (1 electrode split in 2),
      – merging 1–2 triplets of nearby fragments (1 electrode split in 3),
      – any mix of the above (e.g. merge 1 pair + drop 1 noise blob).
    Discarded clusters are penalised by size so the optimiser avoids
    silently discarding large electrode fragments.

  All recovery actions are flagged in the log file.

  Two additional geometry checks are applied to every 4-cluster
  configuration before coordinates are accepted:
    • Too-close check  — any pair of centres closer than 15 mm
      (< 1.5 cm) is rejected as a split/duplicate segmentation.
    • Planarity check  — the 4 centres are projected onto the best-
      fit plane of the star; a cluster that deviates more than
      PLANARITY_THRESHOLD mm out-of-plane is rejected, catching
      stacked segmentations that do not belong to the flat head-
      surface electrode layout.

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
import re
import argparse
import sys
import glob
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from itertools import combinations, permutations

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import center_of_mass, label
from scipy.spatial.distance import euclidean
from scipy.ndimage import measurements

# ============================================================
#  TMP / LOG FOLDER  — created next to this script
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
TMP_DIR    = SCRIPT_DIR / '_tmp'
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ── Network registry ────────────────────────────────────────────────────────
# Maps a network key to the inference-file suffix written by
# Segmentation_paper_models.py, and to the label used in output filenames.
# Keep these in sync with that script: it writes
#     <name>_PDw_inference_<key>.nii.gz
# so the key here IS the suffix. Add a row when a new network is segmented.
NETWORKS = {
    'proposed':       'Proposed Attention U-Net, 32-512',
    'b_no_attention': 'Proposed depths, no attention gates, 32-512',
    'a_baseline':     'Standard U-Net baseline, 64-512',
    'd_increased':    'Attention U-Net, increased widths 48-768',
    'c_reduced':      'Attention U-Net, reduced widths 16-256',
    # legacy suffixes produced by the older segmentation scripts
    'reduced_network':   'legacy: reduced-channel rebuttal run',
    'increased_network': 'legacy: increased-channel rebuttal run',
}

DEFAULT_NETWORK = 'proposed'

# Filled in at start-up from the CLI; used for the log name, the output CSV
# name and the `network` column of the results table.
NETWORK_KEY: str = DEFAULT_NETWORK

LOG_FILE = TMP_DIR / f"log_extract_coordinates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


# ============================================================
#  GEOMETRY THRESHOLDS
# ============================================================

# Minimum allowed centre-to-centre distance between any two electrodes (mm).
# Pairs closer than this are almost certainly split fragments of the same
# physical electrode, not two distinct electrodes.
MIN_INTERELECTRODE_DIST_MM: float = 15.0   # 1.5 cm

# Maximum allowed out-of-plane deviation for the planarity check (mm).
# The four electrodes are expected to lie on or very close to the curved
# scalp surface, which is approximately planar at the scale of the montage.
# A cluster deviating more than this from the best-fit plane is rejected.
PLANARITY_THRESHOLD_MM: float = 20.0   # 2.0 cm (P2 has high curvature)

# ── Electrode geometry thresholds ───────────────────────────────────────────
# Each physical electrode is a disc of ~10 mm radius and 1–4 mm height.
# These bounds are used by check_electrode_size() and try_merge_split_clusters().

# Minimum plausible in-plane radius of a correctly segmented electrode (mm).
# Clusters smaller than this trigger a WARNING but are not rejected.
ELECTRODE_MIN_RADIUS_MM: float = 4.0   # anything < 4 mm radius is suspiciously tiny

# Maximum plausible in-plane radius (mm). Clusters larger than this are
# also flagged — they may be two merged electrodes or a large artefact.
ELECTRODE_MAX_RADIUS_MM: float = 15.0  # > 15 mm radius is suspiciously large

# Nominal expected radius used to judge "two fragments within one electrode
# footprint".  Two cluster centres separated by less than
# ELECTRODE_MERGE_DIST_MM are candidates for merging into a single electrode.
# Set to ~2× the nominal radius so that even off-centre splits are caught.
ELECTRODE_MERGE_DIST_MM: float = 20.0  # 2 cm — two clusters closer than this
                                        # could still be parts of one electrode

# --- Split heuristic (3-cluster recovery: anode/cathode "melted together") ---
# A single electrode is a disc of ~10 mm radius (≤ ~15 mm), i.e. a diameter of
# ~20 mm (≤ ~30 mm). When exactly three clusters are detected, one of them is
# frequently two electrodes that have fused at their touching edges (most often
# the central anode merged with an adjacent cathode). Such a cluster is (i)
# elongated — an oval / figure-eight rather than a circle — and (ii) physically
# too long: its largest in-plane extent is roughly twice a single electrode's
# diameter, while its perpendicular extent stays ~one diameter. The constants
# below define when a 3rd cluster is a split candidate and how the split is
# performed (1-D k-means, k=2, along the cluster's major axis).
#
# Discriminators were calibrated on disc geometry: a single 10–12 mm-radius disc
# has elongation ≈ 1.0 and major in-plane extent ≈ 20–24 mm, whereas two fused
# discs have elongation ≈ 1.8–2.0 and major extent ≈ 36–40 mm.
ELECTRODE_NOMINAL_RADIUS_MM: float = 10.0  # expected single-disc radius

# Elongation = (major in-plane extent) / (minor in-plane extent). Two fused
# discs form an oval/figure-8; a single disc is ~circular (ratio ≈ 1). Require
# at least this ratio so we only split clusters that are genuinely elongated.
SPLIT_MIN_ELONGATION: float = 1.5

# A single electrode's in-plane diameter is ≤ ~30 mm. Require the candidate's
# major extent to exceed this so normal (even slightly large) single discs are
# never split; only a clearly double-length cluster qualifies.
SPLIT_MIN_MAJOR_EXTENT_MM: float = 30.0

# After splitting, each of the two sub-clusters must itself look electrode-sized
# (RMS radius within these bounds) and the two new centres must sit within the
# normal inter-electrode distance band — otherwise the split is rejected and the
# original cluster is kept unchanged.
SPLIT_SUBCLUSTER_MIN_RADIUS_MM: float = 4.0
SPLIT_SUBCLUSTER_MAX_RADIUS_MM: float = 13.0


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
#  FIVE-CLUSTER RECOVERY HELPERS
# ============================================================

def _star_score(centers: list) -> tuple[float, int]:
    """
    Given exactly 4 centre-of-mass coordinates, evaluate how well
    they form a 1-anode / 3-cathode star pattern:
      - one central electrode (anode) surrounded by three cathodes
        at roughly equal angles (≈120°) and roughly equal radii.

    Returns
    -------
    score : float
        Lower is better (0.0 = perfect star).
        Computed as the sum of:
          • radius_cv  — coefficient of variation of the 3 spoke lengths
          • angle_std  — std-dev of the 3 inter-spoke angles (degrees),
                         normalised by 120°
    best_center_idx : int
        Index (0–3) of the electrode identified as the central anode.
    """
    best_score = np.inf
    best_center_idx = 0

    for ci in range(4):
        center = np.array(centers[ci])
        spokes = [np.array(centers[j]) - center for j in range(4) if j != ci]

        radii  = [np.linalg.norm(s) for s in spokes]
        radius_cv = np.std(radii) / (np.mean(radii) + 1e-9)

        # Angles between consecutive spoke pairs (in degrees)
        angles = []
        for i in range(len(spokes)):
            for j in range(i + 1, len(spokes)):
                cos_a = np.dot(spokes[i], spokes[j]) / (
                    np.linalg.norm(spokes[i]) * np.linalg.norm(spokes[j]) + 1e-9
                )
                cos_a = np.clip(cos_a, -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cos_a)))

        angle_std_norm = np.std(angles) / 120.0   # normalise against ideal 120°

        score = radius_cv + angle_std_norm
        if score < best_score:
            best_score = score
            best_center_idx = ci

    return best_score, best_center_idx


# ── Cathode-to-anode distance window ────────────────────────────────────────
# The upper bound was originally 70 mm, the maximum observed in the 54-image
# development cohort. That left no margin: on 4234 images it rejected 247
# extractions, of which 226 lay between 70 and 75 mm -- the clipped tail of a
# normal distribution, not mis-segmentations. Raised once, on the observed
# distribution (a clear gap sits between 120 and 135 mm; everything above that
# is a stray cluster on the far side of the head).
#
# DEFINED ONCE AND USED EVERYWHERE. The recovery paths previously carried their
# own default of 70.0 in their signatures, so changing only the main check left
# 5-cluster and fragment-merge recovery rejecting candidate subsets that the
# main validation would have accepted.
DIST_MIN_MM: float = 5.0
DIST_MAX_MM: float = 125.0

# Why 125 and not 70, 80 or 85: the observed distribution of rejected
# distances over 4234 images was dense from 70 to 95 mm (226 of 284 values in
# 70-75 alone, i.e. the clipped tail of a normal distribution), thinned to
# single counts through 120 mm, and then stopped -- the next value was 136.4,
# followed by 155, 165, 218, 229, 237, 245. Those upper values are
# anatomically impossible for this montage and are stray clusters on the far
# side of the head. The bound therefore sits in the empty interval between
# 119.5 and 136.4 mm, so it is set by where the data stops rather than by a
# round number chosen near the body of the distribution.
#
# CAVEAT, and the reason DIST_AUDIT_TIERS exists: extractions between roughly
# 90 and 120 mm are admitted by this bound but are far outside the 30-70 mm
# range typical of this montage. They are few, and they have NOT been verified
# visually. The audit below lists them by name at every run so they can be
# checked rather than assumed.
DIST_AUDIT_TIERS: tuple = (70.0, 80.0, 90.0, 100.0, 110.0, 120.0)

# Every accepted image whose largest spoke exceeds this is named individually
# in the summary (the others are only counted).
DIST_AUDIT_NAME_ABOVE: float = 90.0


# ── Star-geometry validation ────────────────────────────────────────────────
# The montage is a 3x1 star: one central anode with three cathodes at 120deg
# azimuth. It sits on a CURVED scalp, so the straight-line angles between
# spokes are NOT 120deg -- on an 85 mm head with 70 mm spacing they are ~105deg,
# and they shrink further on smaller heads. What curvature preserves is the
# SYMMETRY: all three angles stay equal to each other. So the test must be
# about symmetry, never about a literal 120deg.
#
# NOTE ON _star_score: it sums radius_cv and angle_std/120. That sum is exactly
# 0 for a perfect montage on a SPHERE, but 0.13-0.20 on a realistic ELLIPSOID,
# which is more than some genuinely wrong configurations score (3 cathodes
# bunched 30deg apart scores 0.11). A threshold on the sum would therefore
# reject valid extractions from elongated heads while admitting bunched ones.
# The two components are thresholded separately instead:
#
#   min_angle  catches bunching       (valid >~90deg; bunched 76deg; square 45deg)
#   radius_cv  catches unequal spokes (valid <~0.22; one spoke 2x long 0.32)
#
# Both default to None = MEASURE ONLY: the values are computed, written to the
# CSV and summarised, but nothing is rejected. Set them from the observed
# distribution once you have looked at it -- not from these simulated figures.
STAR_MIN_ANGLE_DEG: float | None = None     # e.g. 80.0 once calibrated
STAR_MAX_RADIUS_CV: float | None = None     # e.g. 0.30 once calibrated


def star_metrics(centers: list) -> tuple[int, float, float, float]:
    """Hub index plus the three symmetry features, curvature-safe.

    The hub is the point maximising the MINIMUM inter-spoke angle, which is
    more robust than minimising _star_score: on an ellipsoid the summed score
    can prefer a wrong hub, whereas the widest-spread interpretation is the
    one that actually looks like a star.

    Returns (hub_idx, radius_cv, min_angle_deg, angle_std_deg).
    """
    best = None
    for hub in range(4):
        c = np.array(centers[hub], dtype=float)
        spokes = [np.array(centers[j], dtype=float) - c
                  for j in range(4) if j != hub]
        radii = [float(np.linalg.norm(s)) for s in spokes]
        if min(radii) < 1e-6:
            continue
        radius_cv = float(np.std(radii) / (np.mean(radii) + 1e-9))
        angles = []
        for i in range(3):
            for j in range(i + 1, 3):
                cos_a = np.dot(spokes[i], spokes[j]) / (
                    np.linalg.norm(spokes[i]) * np.linalg.norm(spokes[j]) + 1e-9)
                angles.append(float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))))
        cand = (hub, radius_cv, min(angles), float(np.std(angles)))
        if best is None or cand[2] > best[2]:
            best = cand
    return best if best is not None else (0, 0.0, 0.0, 0.0)


def _is_valid_star(centers: list,
                   dist_min: float = DIST_MIN_MM,
                   dist_max: float = DIST_MAX_MM) -> bool:
    """
    Quick check: does this set of 4 centres pass the existing
    distance-range validation used in is_valid_configuration?
    """
    for perm in permutations(centers):
        distances = [euclidean(perm[0], s) for s in perm[1:]]
        if all(dist_min <= d <= dist_max for d in distances):
            return True
    return False


def filter_fifth_cluster(clusters: dict,
                          score_threshold: float = 0.35,
                          dist_min: float = DIST_MIN_MM,
                          dist_max: float = DIST_MAX_MM) -> tuple[dict | None, int | None, str]:
    """
    When exactly 5 clusters are present, attempt to identify and
    remove the spurious one so that a valid 4-electrode star remains.

    Strategy (applied in order; first success wins):

    1. **Distance outlier** — compute the mean pairwise distance among
       all 5 centres.  If one electrode is more than 2× that mean away
       from every other electrode, it is the outlier.

    2. **Star-fit exhaustive search** — for every possible subset of 4
       clusters, score the star quality with `_star_score`.  Accept the
       best-scoring subset if its score is below `score_threshold` AND
       it passes the distance-range check (`_is_valid_star`).

    Parameters
    ----------
    clusters        : dict returned by find_electrode_clusters (5 entries)
    score_threshold : maximum acceptable star score (default 0.35)
    dist_min/max    : forwarded to _is_valid_star

    Returns
    -------
    filtered_clusters : dict with 4 entries, or None if recovery failed
    removed_key       : original cluster key that was dropped, or None
    reason            : human-readable string describing what was done
    """
    keys    = list(clusters.keys())
    centers = [np.array(clusters[k]['center_of_mass']) for k in keys]

    # ── Strategy 1: distance outlier ────────────────────────────────────
    # Compute average distance of each electrode to all others
    n = len(centers)
    avg_dists = []
    for i in range(n):
        dists = [euclidean(centers[i], centers[j]) for j in range(n) if j != i]
        avg_dists.append(np.mean(dists))

    global_mean = np.mean(avg_dists)
    outlier_idx = int(np.argmax(avg_dists))

    if avg_dists[outlier_idx] > 2.0 * global_mean:
        candidate_keys    = [keys[i] for i in range(n) if i != outlier_idx]
        candidate_centers = [centers[i] for i in range(n) if i != outlier_idx]

        if _is_valid_star(candidate_centers, dist_min, dist_max):
            filtered = {k: clusters[k] for k in candidate_keys}
            reason   = (
                f"Distance-outlier rule: cluster key={keys[outlier_idx]} "
                f"had mean-dist={avg_dists[outlier_idx]:.1f} mm vs "
                f"global mean={global_mean:.1f} mm (ratio "
                f"{avg_dists[outlier_idx]/global_mean:.2f} > 2.0)."
            )
            return filtered, keys[outlier_idx], reason

    # ── Strategy 2: best star-fit over all 4-of-5 subsets ───────────────
    best_score   = np.inf
    best_subset  = None
    removed_key  = None

    for drop_i, drop_key in enumerate(keys):
        subset_keys    = [keys[i] for i in range(n) if i != drop_i]
        subset_centers = [centers[i] for i in range(n) if i != drop_i]

        if not _is_valid_star(subset_centers, dist_min, dist_max):
            continue

        score, _ = _star_score(subset_centers)
        if score < best_score:
            best_score  = score
            best_subset = subset_keys
            removed_key = drop_key

    if best_score <= score_threshold and best_subset is not None:
        filtered = {k: clusters[k] for k in best_subset}
        reason   = (
            f"Star-fit rule: dropped cluster key={removed_key} "
            f"(star score={best_score:.4f} ≤ threshold={score_threshold})."
        )
        return filtered, removed_key, reason

    # ── Recovery failed ──────────────────────────────────────────────────
    return None, None, (
        f"Could not identify a spurious 5th cluster "
        f"(best star score={best_score:.4f} > threshold={score_threshold})."
    )


# ============================================================
#  ADDITIONAL GEOMETRY CHECKS FOR 4-CLUSTER CONFIGURATIONS
# ============================================================

def check_min_distance(clusters: dict,
                       min_dist_mm: float = MIN_INTERELECTRODE_DIST_MM
                       ) -> tuple[bool, str]:
    """
    Reject a 4-cluster set if any pair of centres is closer than
    `min_dist_mm` millimetres.

    This catches cases where the segmentation has split a single
    physical electrode into two nearby fragments, producing a
    spurious "4-electrode" result that would otherwise pass the
    count check.

    Parameters
    ----------
    clusters    : dict of cluster dicts (must contain 'center_of_mass')
    min_dist_mm : minimum allowed pairwise distance in mm (default 15 mm)

    Returns
    -------
    ok     : True if all pairs are at least min_dist_mm apart
    reason : human-readable failure message (empty string if ok)
    """
    keys    = list(clusters.keys())
    centers = [np.array(clusters[k]['center_of_mass']) for k in keys]

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = euclidean(centers[i], centers[j])
            if d < min_dist_mm:
                return False, (
                    f"Too-close pair: cluster keys {keys[i]} and {keys[j]} "
                    f"are only {d:.1f} mm apart (threshold {min_dist_mm:.0f} mm). "
                    f"Likely a split segmentation fragment."
                )
    return True, ""


def check_planarity(clusters: dict,
                    threshold_mm: float = PLANARITY_THRESHOLD_MM
                    ) -> tuple[bool, str]:
    """
    Reject a 4-cluster set if any centre deviates more than
    `threshold_mm` millimetres from the best-fit plane of all
    four centres.

    The four head-surface electrodes form a quasi-planar star on
    the scalp. A segmentation that was picked up in a different
    anatomical plane (e.g. a cluster directly above/below the
    montage rather than beside it) will fail this test.

    Method
    ------
    1. Compute the centroid of the four centres.
    2. Build the 3×4 matrix of centred coordinates.
    3. The normal to the best-fit plane is the singular vector
       corresponding to the *smallest* singular value (standard
       PCA / SVD approach).
    4. The out-of-plane deviation of each point is its projection
       onto that normal.

    Parameters
    ----------
    clusters     : dict of cluster dicts (must contain 'center_of_mass')
    threshold_mm : maximum allowed out-of-plane deviation in mm (default 20 mm, because P2 shows high curvature)

    Returns
    -------
    ok     : True if all four centres are within the planar tolerance
    reason : human-readable failure message (empty string if ok)
    """
    keys    = list(clusters.keys())
    centers = np.array([clusters[k]['center_of_mass'] for k in keys])   # (4, 3)

    centroid = centers.mean(axis=0)
    centered = centers - centroid                                         # (4, 3)

    # SVD: right singular vectors are the principal axes.
    # The last one (V[2]) is the plane normal.
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[2]                                                        # unit vector

    # Signed out-of-plane distances
    deviations = np.abs(centered @ normal)                                # (4,)

    worst_idx = int(np.argmax(deviations))
    worst_dev = deviations[worst_idx]

    if worst_dev > threshold_mm:
        return False, (
            f"Planarity check failed: cluster key={keys[worst_idx]} "
            f"deviates {worst_dev:.1f} mm out of the best-fit plane "
            f"(threshold {threshold_mm:.0f} mm). "
            f"Likely a stacked/off-plane segmentation fragment."
        )
    return True, ""


# ============================================================
#  ELECTRODE-SIZE SANITY CHECK  (warning-only, never rejects)
# ============================================================

def check_electrode_size(clusters: dict,
                         affine: np.ndarray,
                         min_radius_mm: float = ELECTRODE_MIN_RADIUS_MM,
                         max_radius_mm: float = ELECTRODE_MAX_RADIUS_MM,
                         ) -> list[str]:
    """
    Inspect the physical dimensions of every cluster and emit a warning
    string for each one whose effective radius is outside the expected
    range for a scalp electrode (~10 mm radius, 1–4 mm height).

    This function is purely advisory — it never rejects a cluster.
    The caller is responsible for logging the returned warnings.

    Method
    ------
    For each cluster the voxel coordinates are converted to world-space
    (mm) using the image affine.  The effective in-plane radius is
    estimated as the square root of the projected area divided by π,
    where the projection plane is determined by PCA (smallest variance
    axis ≈ electrode thickness axis).

    Parameters
    ----------
    clusters      : dict produced by find_electrode_clusters
    affine        : 4×4 NIfTI affine (voxel → mm)
    min_radius_mm : warn if effective radius is below this value (mm)
    max_radius_mm : warn if effective radius is above this value (mm)

    Returns
    -------
    warnings : list of human-readable warning strings (empty if all ok)
    """
    warnings_out = []

    for key, cl in clusters.items():
        coords_vox = cl['coords']           # (N, 3) voxel coordinates

        if len(coords_vox) == 0:
            warnings_out.append(
                f"Cluster key={key}: no voxels — cannot estimate size."
            )
            continue

        # Convert voxel coordinates to world (mm) coordinates
        coords_mm = nib.affines.apply_affine(affine, coords_vox)  # (N, 3)

        if len(coords_mm) < 4:
            # Degenerate cluster — too few voxels for PCA
            warnings_out.append(
                f"Cluster key={key}: only {len(coords_mm)} voxel(s) — "
                f"likely a noise fragment (expected ≥ hundreds of voxels)."
            )
            continue

        # PCA to find the thickness axis (axis of smallest variance)
        centroid_mm = coords_mm.mean(axis=0)
        centered_mm = coords_mm - centroid_mm                       # (N, 3)
        _, _, Vt    = np.linalg.svd(centered_mm, full_matrices=False)
        # Vt[2] is the direction of least variance ≈ electrode normal
        thickness_axis = Vt[2]

        # Project all points onto the plane perpendicular to thickness_axis
        # (i.e. remove the thickness component)
        projections = centered_mm - np.outer(
            centered_mm @ thickness_axis, thickness_axis
        )                                                            # (N, 3)

        # Effective radius = sqrt(mean squared in-plane distance from centroid)
        # This is the RMS radius, a robust proxy for disc radius.
        rms_radius = np.sqrt(np.mean(np.sum(projections ** 2, axis=1)))

        if rms_radius < min_radius_mm:
            warnings_out.append(
                f"Cluster key={key}: effective radius ≈ {rms_radius:.1f} mm "
                f"is BELOW the expected minimum of {min_radius_mm:.0f} mm. "
                f"The cluster may be a small segmentation fragment, not a "
                f"complete electrode disc."
            )
        elif rms_radius > max_radius_mm:
            warnings_out.append(
                f"Cluster key={key}: effective radius ≈ {rms_radius:.1f} mm "
                f"EXCEEDS the expected maximum of {max_radius_mm:.0f} mm. "
                f"The cluster may be a merged artefact or include surrounding "
                f"tissue."
            )

    return warnings_out


# ============================================================
#  GENERAL MULTI-CLUSTER RECOVERY  (n = 5 … MAX_RECOVERY_CLUSTERS)
# ============================================================

# Hard upper limit: images with more clusters than this are considered
# too fragmented to recover reliably and are marked invalid immediately.
MAX_RECOVERY_CLUSTERS: int = 8

# Star-score acceptance threshold for the general recovery function.
# Only the *pure* star score (not including any discard penalty) is
# compared against this.  The value 0.50 is intentionally a little more
# generous than filter_fifth_cluster's 0.35 because the merged CoM of
# two fragments is noisier than a clean single-cluster centre.
MULTI_CLUSTER_STAR_SCORE_THRESHOLD: float = 0.50

# Penalty added to the combined score for each discarded cluster,
# normalised by the mean cluster size.  A discarded cluster whose size
# equals the mean cluster size contributes exactly this much to the
# score, steering the optimiser away from solutions that silently throw
# away large (potentially real) electrode fragments.
MULTI_CLUSTER_DISCARD_PENALTY: float = 0.30


def recover_four_electrodes(
    clusters: dict,
    affine: np.ndarray,
    merge_dist_mm: float = ELECTRODE_MERGE_DIST_MM,
    min_interelectrode_mm: float = MIN_INTERELECTRODE_DIST_MM,
    max_group_size: int = 3,
    star_score_threshold: float = MULTI_CLUSTER_STAR_SCORE_THRESHOLD,
    discard_penalty: float = MULTI_CLUSTER_DISCARD_PENALTY,
) -> tuple[dict | None, str]:
    """
    General recovery: find the best partition of n clusters into exactly
    4 "electrode groups", handling every combination of fragment merging
    and spurious-cluster removal.

    This supersedes `try_merge_split_clusters` and covers all realistic
    scenarios that produce n > 4 clusters after size-based filtering:

    ┌──────┬─────────────────────────────────────────────────────────────┐
    │  n   │  Scenarios handled                                          │
    ├──────┼─────────────────────────────────────────────────────────────┤
    │  5   │  drop 1 noise blob                                          │
    │      │  merge 1 pair  (1 electrode split into 2 fragments)        │
    ├──────┼─────────────────────────────────────────────────────────────┤
    │  6   │  drop 2 noise blobs                                         │
    │      │  merge 2 pairs (2 electrodes each split into 2 fragments)  │
    │      │  merge 1 triplet (1 electrode split into 3 fragments)      │
    │      │  merge 1 pair + drop 1 noise blob                          │
    ├──────┼─────────────────────────────────────────────────────────────┤
    │  7   │  drop 3 noise blobs                                         │
    │      │  merge 3 pairs                                              │
    │      │  merge 2 pairs + drop 1 noise blob                         │
    │      │  merge 1 triplet + merge 1 pair                             │
    │      │  merge 1 triplet + drop 1 noise blob                       │
    │      │  merge 1 pair   + drop 2 noise blobs                       │
    ├──────┼─────────────────────────────────────────────────────────────┤
    │  8   │  all of the above extended by one more operation            │
    └──────┴─────────────────────────────────────────────────────────────┘

    Algorithm
    ---------
    1. **Build candidate electrode groups** — every subset of 1 to
       `max_group_size` cluster keys where the *maximum* pairwise
       world-space distance within the subset is ≤ `merge_dist_mm`.

       • size-1 group : single cluster, kept as-is (no merge).
       • size-2 group : pair of nearby fragments → merged into 1 electrode.
       • size-3 group : triplet of nearby fragments → merged into 1 electrode.
         All three pairwise distances must each be ≤ merge_dist_mm (i.e. they
         are mutually close, not merely connected in a chain).

    2. **Backtracking search** — enumerate every combination of exactly 4
       *disjoint* candidate groups (no cluster key appears in two groups).
       Clusters assigned to no group are silently discarded.  Using a
       monotone index ensures each unordered combination is visited once.

    3. **Scoring** — each candidate solution receives a combined score:

           total_score = star_score
                       + Σ_discarded (discard_penalty × size_k / mean_size)

       The star score (from _star_score) measures how well the 4 merged
       centres form a 1-anode / 3-cathode equidistant star.  The discard
       term penalises throwing away large clusters that are likely real
       electrode fragments.  The optimiser therefore prefers solutions that
       (a) form a good star AND (b) explain as many clusters as possible.

    4. **Acceptance** — the best solution is accepted if its *pure star
       score* (without the discard penalty) is ≤ `star_score_threshold`
       AND every pair of the 4 merged centres is at least
       `min_interelectrode_mm` apart AND the centres pass the star
       distance-range check (_is_valid_star).

    Merged cluster properties
    -------------------------
    CoM — size-weighted average of all fragment CoMs in the group:

        CoM_merged = Σ (size_i × CoM_i) / Σ size_i

    This is the true CoM of the voxel union and gives the best estimate
    of the physical electrode centre regardless of how the fragments split.

    coords       — vstack of all fragment coordinate arrays.
    size         — sum of all fragment sizes.
    merged_from  — sorted list of the original cluster keys in the group.

    Parameters
    ----------
    clusters              : dict from find_electrode_clusters (n entries)
    affine                : 4×4 NIfTI affine (voxel → mm)
    merge_dist_mm         : max pairwise world-space distance (mm) for two
                            clusters to be considered part of one electrode
    min_interelectrode_mm : min world-space distance (mm) between any two of
                            the 4 final electrode centres after merging
    max_group_size        : max number of fragments merged into one electrode
                            (default 3; set to 2 to disable triplet merging)
    star_score_threshold  : max acceptable pure star score (default 0.50)
    discard_penalty       : per-cluster score penalty for discarded clusters,
                            normalised by mean cluster size (default 0.30)

    Returns
    -------
    merged_clusters : dict with exactly 4 entries, or None on failure.
                      Each entry may carry a 'merged_from' key listing
                      which original clusters were combined.
    reason          : human-readable description of the outcome.
    """
    keys = list(clusters.keys())
    n    = len(keys)

    if n <= 4:
        return None, (
            f"recover_four_electrodes requires n > 4 clusters; "
            f"got {n} — not applicable."
        )
    if n > MAX_RECOVERY_CLUSTERS:
        return None, (
            f"Cluster count {n} exceeds the recovery limit "
            f"({MAX_RECOVERY_CLUSTERS}).  Image appears too fragmented "
            f"for reliable automatic recovery."
        )

    # ── Pre-compute world-space CoMs and sizes once ─────────────────────────
    def _com_mm(cl: dict) -> np.ndarray:
        return nib.affines.apply_affine(affine, np.array(cl['center_of_mass']))

    coms_mm   = {k: _com_mm(clusters[k])                                        for k in keys}
    sizes     = {k: float(clusters[k].get('size', len(clusters[k]['coords'])))  for k in keys}
    mean_size = float(np.mean(list(sizes.values()))) + 1e-9  # avoid /0

    # ── Build candidate electrode groups ────────────────────────────────────
    # Each candidate is a frozenset of cluster keys that could all belong
    # to the same physical electrode.
    #
    # Proximity rule: EVERY pairwise distance within the subset must be
    # ≤ merge_dist_mm.  For size-2 that is one check; for size-3 it is
    # three checks, ensuring the three fragments are mutually close rather
    # than forming a chain (A–B close, B–C close but A–C far apart would
    # span two electrode footprints and is therefore excluded).

    candidate_groups: list[frozenset] = []

    # Size-1: every individual cluster is always a candidate
    for k in keys:
        candidate_groups.append(frozenset([k]))

    # Size-2: pairs whose world-space distance ≤ merge_dist_mm
    for i in range(n):
        for j in range(i + 1, n):
            ki, kj = keys[i], keys[j]
            if euclidean(coms_mm[ki], coms_mm[kj]) <= merge_dist_mm:
                candidate_groups.append(frozenset([ki, kj]))

    # Size-3: triples where ALL three pairwise distances ≤ merge_dist_mm
    if max_group_size >= 3:
        for i in range(n):
            for j in range(i + 1, n):
                for kk in range(j + 1, n):
                    ki, kj, kkk = keys[i], keys[j], keys[kk]
                    d_ij = euclidean(coms_mm[ki],  coms_mm[kj])
                    d_ik = euclidean(coms_mm[ki],  coms_mm[kkk])
                    d_jk = euclidean(coms_mm[kj],  coms_mm[kkk])
                    if max(d_ij, d_ik, d_jk) <= merge_dist_mm:
                        candidate_groups.append(frozenset([ki, kj, kkk]))

    if len(candidate_groups) < 4:
        return None, (
            f"Only {len(candidate_groups)} candidate group(s) found for "
            f"{n} clusters — not enough to form 4 electrode groups."
        )

    # ── Helper: size-weighted merge of a group ──────────────────────────────
    def _merge_group(group_keys: frozenset) -> dict:
        """Return a new cluster dict representing the union of all keys."""
        total_size = sum(sizes[k] for k in group_keys)
        com_vox    = (
            sum(sizes[k] * np.array(clusters[k]['center_of_mass'])
                for k in group_keys)
            / total_size
        )
        all_coords = np.vstack([clusters[k]['coords'] for k in group_keys])
        return {
            'coords':          all_coords,
            'center_of_mass':  tuple(float(x) for x in com_vox),
            'size':            total_size,
            'merged_from':     sorted(group_keys),
        }

    # ── Score a candidate solution ───────────────────────────────────────────
    def _score_solution(
        four_groups: list[frozenset],
    ) -> tuple[float, float, dict | None]:
        """
        Evaluate a candidate set of 4 disjoint groups.

        Returns
        -------
        (total_score, star_score, merged_dict)
        Returns (inf, inf, None) when the configuration fails geometry checks.
        """
        # Build the 4 merged electrode clusters
        merged  = {}
        new_key = max(keys) + 1
        for g in four_groups:
            merged[new_key] = _merge_group(g)
            new_key        += 1

        # ── Geometry check 1: minimum inter-electrode distance ───────────
        mk_list    = list(merged.keys())
        coms_world = [_com_mm(merged[k]) for k in mk_list]
        for i in range(4):
            for j in range(i + 1, 4):
                if euclidean(coms_world[i], coms_world[j]) < min_interelectrode_mm:
                    return np.inf, np.inf, None

        # ── Geometry check 2: star distance-range (voxel space) ──────────
        # Uses voxel-space CoMs for consistency with is_valid_configuration.
        vox_coms = [merged[k]['center_of_mass'] for k in mk_list]
        if not _is_valid_star(vox_coms):
            return np.inf, np.inf, None

        # ── Star quality score ────────────────────────────────────────────
        star_sc, _ = _star_score(vox_coms)

        # ── Discard penalty ───────────────────────────────────────────────
        covered   = frozenset().union(*four_groups)
        disc_keys = [k for k in keys if k not in covered]
        penalty   = discard_penalty * sum(
            sizes[k] / mean_size for k in disc_keys
        )

        return star_sc + penalty, star_sc, merged

    # ── Backtracking search for the best 4 disjoint groups ──────────────────
    # Enumerate all combinations of 4 disjoint candidate groups by advancing
    # a monotone start index so each unordered combination is visited once.
    # Early pruning: abort when fewer candidates remain than slots needed.

    best: dict = {'total': np.inf, 'star': np.inf, 'merged': None, 'groups': None}

    def _search(start: int, used: frozenset, chosen: list) -> None:
        if len(chosen) == 4:
            total_sc, star_sc, merged = _score_solution(chosen)
            if total_sc < best['total']:
                best['total']  = total_sc
                best['star']   = star_sc
                best['merged'] = merged
                best['groups'] = chosen[:]
            return

        slots_left = 4 - len(chosen)
        if len(candidate_groups) - start < slots_left:
            return   # not enough candidates left — prune branch

        for idx in range(start, len(candidate_groups)):
            grp = candidate_groups[idx]
            if not grp.isdisjoint(used):
                continue   # cluster key already assigned — skip
            _search(idx + 1, used | grp, chosen + [grp])

    _search(0, frozenset(), [])

    # ── Evaluate and return ─────────────────────────────────────────────────
    if best['merged'] is None:
        return None, (
            f"No valid 4-electrode configuration found among "
            f"{len(candidate_groups)} candidate groups ({n} input clusters)."
        )

    if best['star'] > star_score_threshold:
        return None, (
            f"Best recovered star score {best['star']:.4f} exceeds "
            f"threshold {star_score_threshold:.2f} — all candidate "
            f"configurations rejected ({n} input clusters)."
        )

    # ── Build human-readable summary ─────────────────────────────────────────
    covered  = frozenset().union(*best['groups'])
    disc     = [k for k in keys if k not in covered]

    group_parts = []
    for g in best['groups']:
        g_sorted = sorted(g)
        if len(g_sorted) == 1:
            group_parts.append(f"key {g_sorted[0]} kept as-is")
        else:
            dist_strs = []
            for ii in range(len(g_sorted)):
                for jj in range(ii + 1, len(g_sorted)):
                    d = euclidean(coms_mm[g_sorted[ii]], coms_mm[g_sorted[jj]])
                    dist_strs.append(
                        f"{g_sorted[ii]}↔{g_sorted[jj]}: {d:.1f} mm"
                    )
            tag = "pair" if len(g_sorted) == 2 else "triplet"
            group_parts.append(
                f"keys [{'+'.join(str(k) for k in g_sorted)}] merged as {tag} "
                f"({', '.join(dist_strs)})"
            )

    reason_lines = [
        f"General recovery {n}→4 clusters  |  "
        f"star score={best['star']:.4f}, total score={best['total']:.4f}",
        "Groups: " + ";  ".join(group_parts),
    ]
    if disc:
        disc_detail = ", ".join(
            f"key {k} (size={int(sizes[k])})" for k in sorted(disc)
        )
        reason_lines.append(f"Discarded: {disc_detail}")

    return best['merged'], "  |  ".join(reason_lines)



# ============================================================
#  CORE FUNCTIONS
# ============================================================

def find_nifti_files(base_path, network_key: str = None):
    """Return the inference volumes belonging to ONE network.

    The suffix is what Segmentation_paper_models.py appended, i.e.
        <name>_PDw_inference_<network_key>.nii.gz
    Restricting the glob to a single suffix is what keeps the networks'
    coordinate tables separate: without it, a directory holding several
    networks' segmentations would silently mix them into one table.
    """
    key = network_key or NETWORK_KEY
    pattern = os.path.join(
        base_path, "sub-*", "unzipped", f"*_inference_{key}.nii.gz"
    )
    return sorted(glob.glob(pattern))


def voxel_to_corr_space(voxel_coords, affine):
    return nib.affines.apply_affine(affine, voxel_coords)


def _cluster_inplane_geometry(coords_vox, affine):
    """
    Describe a cluster's flat (disc-like) geometry in world (mm) space.

    The electrode is a thin disc, so its voxels lie close to a plane. We use
    PCA: the two largest-variance axes span the disc plane, the smallest is the
    thickness/normal. Returns the quantities needed to decide whether a cluster
    is one disc or two fused discs, and to split it if necessary.

    Returns a dict with:
      centroid_mm    : (3,) world centroid
      plane_axes     : (2, 3) the two in-plane PCA directions (major, minor)
      normal_axis    : (3,)   the thickness direction
      major_extent   : full extent (max-min projection) along the major axis, mm
      minor_extent   : full extent along the minor in-plane axis, mm
      rms_radius     : RMS in-plane radius, mm (same proxy as check_electrode_size)
      elongation     : major_extent / minor_extent  (≈1 for a disc)
      coords_mm      : (N, 3) world coordinates
    Returns None if the cluster has too few voxels for a stable PCA.
    """
    coords_mm = nib.affines.apply_affine(affine, coords_vox)
    if len(coords_mm) < 8:
        return None

    centroid_mm = coords_mm.mean(axis=0)
    centered    = coords_mm - centroid_mm
    _, _, Vt    = np.linalg.svd(centered, full_matrices=False)
    major_axis, minor_axis, normal_axis = Vt[0], Vt[1], Vt[2]

    proj_major = centered @ major_axis
    proj_minor = centered @ minor_axis

    major_extent = float(proj_major.max() - proj_major.min())
    minor_extent = float(proj_minor.max() - proj_minor.min())

    # in-plane RMS radius (drop the thickness component), matching the
    # convention used by check_electrode_size()
    inplane = centered - np.outer(centered @ normal_axis, normal_axis)
    rms_radius = float(np.sqrt(np.mean(np.sum(inplane ** 2, axis=1))))

    elong = major_extent / minor_extent if minor_extent > 1e-6 else np.inf

    return {
        'centroid_mm':  centroid_mm,
        'plane_axes':   np.vstack([major_axis, minor_axis]),
        'normal_axis':  normal_axis,
        'major_extent': major_extent,
        'minor_extent': minor_extent,
        'rms_radius':   rms_radius,
        'elongation':   elong,
        'coords_mm':    coords_mm,
    }


def _split_cluster_in_two(cl, affine, geom):
    """
    Split one voxel cluster into two along its major in-plane axis.

    Uses a simple 1-D, 2-means partition of the major-axis projection (no
    sklearn dependency). Returns two new cluster dicts in the same format as
    find_electrode_clusters (keys: coords, center_of_mass, size), or None if
    either side is empty.
    """
    coords_vox = cl['coords']
    coords_mm  = geom['coords_mm']
    centroid   = geom['centroid_mm']
    major      = geom['plane_axes'][0]

    t = (coords_mm - centroid) @ major          # 1-D coordinate along major axis

    # 1-D k-means (k=2) initialised at the extremes.
    c0, c1 = t.min(), t.max()
    for _ in range(50):
        left = np.abs(t - c0) <= np.abs(t - c1)
        if left.all() or (~left).all():
            break
        nc0, nc1 = t[left].mean(), t[~left].mean()
        if np.isclose(nc0, c0) and np.isclose(nc1, c1):
            break
        c0, c1 = nc0, nc1

    right = ~left
    if left.sum() == 0 or right.sum() == 0:
        return None

    out = []
    for mask in (left, right):
        sub_vox = coords_vox[mask]
        out.append({
            'coords':         sub_vox,
            'center_of_mass': tuple(sub_vox.mean(axis=0)),
            'size':           float(mask.sum()),
        })
    return out[0], out[1]


def try_split_merged_cluster(clusters, affine):
    """
    Recover a 4-electrode configuration from exactly THREE detected clusters.

    Rationale
    ---------
    A scalp electrode is a thin disc of ~10 mm radius (≤ ~15 mm). When the
    network outputs only three clusters, the most common cause (confirmed by
    visual QC of the rebuttal cohort) is that two neighbouring electrodes —
    typically the central anode and one cathode — have fused at their touching
    edges and are labelled as a single connected component. Such a cluster is
    both oversized (radius well above one disc) and elongated (an oval / figure-
    eight rather than a circle).

    Strategy
    --------
    1. Only act when there are exactly 3 clusters.
    2. Find the most split-like cluster: in-plane RMS radius
       >= SPLIT_MIN_RMS_RADIUS_MM AND elongation >= SPLIT_MIN_ELONGATION.
       (Both conditions guard against splitting a merely-large single disc.)
    3. Split that cluster in two along its major axis (1-D 2-means).
    4. Accept the split only if BOTH halves are electrode-sized
       (SPLIT_SUBCLUSTER_MIN/MAX_RADIUS_MM) and their centres are no closer
       than MIN_INTERELECTRODE_DIST_MM (so we never manufacture a too-close
       pair). Otherwise reject and leave the input unchanged.

    Parameters
    ----------
    clusters : dict from find_electrode_clusters (exactly 3 entries expected)
    affine   : 4×4 NIfTI affine (voxel → mm)

    Returns
    -------
    (new_clusters, reason) : new_clusters is a 4-entry dict on success, or None
                             on failure. reason is a human-readable log string.
    """
    if len(clusters) != 3:
        return None, f"Split skipped: expected 3 clusters, got {len(clusters)}."

    keys = list(clusters.keys())

    # Score each cluster for "looks like two fused discs".
    best_key, best_geom, best_score = None, None, -np.inf
    geoms = {}
    for k in keys:
        geom = _cluster_inplane_geometry(clusters[k]['coords'], affine)
        geoms[k] = geom
        if geom is None:
            continue
        if (geom['major_extent'] >= SPLIT_MIN_MAJOR_EXTENT_MM and
                geom['elongation'] >= SPLIT_MIN_ELONGATION):
            # prefer the most elongated, then the longest
            score = geom['elongation'] + geom['major_extent'] / 100.0
            if score > best_score:
                best_key, best_geom, best_score = k, geom, score

    if best_key is None:
        return None, (
            "Split skipped: no cluster met the merged-electrode criteria "
            f"(need major extent >= {SPLIT_MIN_MAJOR_EXTENT_MM:.0f} mm and "
            f"elongation >= {SPLIT_MIN_ELONGATION:.1f}). "
            "Likely a genuinely missing electrode, not a fused pair.")

    split = _split_cluster_in_two(clusters[best_key], affine, best_geom)
    if split is None:
        return None, f"Split failed: could not partition cluster key={best_key}."
    subA, subB = split

    # Validate the two halves.
    gA = _cluster_inplane_geometry(subA['coords'], affine)
    gB = _cluster_inplane_geometry(subB['coords'], affine)
    if gA is None or gB is None:
        return None, (f"Split rejected: a half of cluster key={best_key} was "
                      "too small for a stable size estimate.")

    for tag_, g in (('A', gA), ('B', gB)):
        if not (SPLIT_SUBCLUSTER_MIN_RADIUS_MM <= g['rms_radius']
                <= SPLIT_SUBCLUSTER_MAX_RADIUS_MM):
            return None, (
                f"Split rejected: half {tag_} of cluster key={best_key} has "
                f"radius {g['rms_radius']:.1f} mm, outside the single-electrode "
                f"range [{SPLIT_SUBCLUSTER_MIN_RADIUS_MM:.0f}, "
                f"{SPLIT_SUBCLUSTER_MAX_RADIUS_MM:.0f}] mm.")

    sep = float(np.linalg.norm(gA['centroid_mm'] - gB['centroid_mm']))
    if sep < MIN_INTERELECTRODE_DIST_MM:
        return None, (
            f"Split rejected: the two halves of cluster key={best_key} are only "
            f"{sep:.1f} mm apart (< {MIN_INTERELECTRODE_DIST_MM:.0f} mm); this "
            "would create a too-close pair.")

    # Build the new 4-cluster dict: the two untouched clusters + the two halves.
    new_clusters = {k: clusters[k] for k in keys if k != best_key}
    next_key = max(keys) + 1
    new_clusters[best_key] = subA
    new_clusters[next_key] = subB

    reason = (
        f"3-CLUSTER SPLIT: cluster key={best_key} "
        f"(major extent {best_geom['major_extent']:.1f} mm, elongation "
        f"{best_geom['elongation']:.2f}) split into two electrode-sized discs "
        f"({gA['rms_radius']:.1f} mm and {gB['rms_radius']:.1f} mm radius, "
        f"{sep:.1f} mm apart). Interpreted as two electrodes fused at the edge.")
    return new_clusters, reason


def _cluster_volumes_mm3(cluster_list, affine) -> list:
    """Voxel count of each cluster converted to mm^3.

    The four electrodes are physically identical objects, so on a correct
    extraction these four numbers should be similar. They are not a proxy for
    Dice -- two masks with the same volume can sit in different places -- but
    they carry information Dice cannot: whether the network systematically
    over- or under-segments, and whether a cluster is roughly twice
    electrode-sized, which is what a fused pair looks like.
    """
    vox_mm3 = float(abs(np.linalg.det(np.asarray(affine)[:3, :3])))
    out = []
    for c in cluster_list:
        n = c.get('size')
        if n is None:
            n = len(c['coords'])
        out.append(float(n) * vox_mm3)
    return out


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
                'size':           float(np.sum(cluster_mask)),
            }

    return clusters


def is_valid_configuration(clusters, counters, subject, session, run):
    tag = f'{subject}_{session}_{run}'

    # ── Count check ─────────────────────────────────────────
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

    # ── Too-close check ─────────────────────────────────────
    # Reject if any pair of centres is less than MIN_INTERELECTRODE_DIST_MM
    # apart (likely a split segmentation fragment, not two real electrodes).
    dist_ok, dist_reason = check_min_distance(clusters)
    if not dist_ok:
        counters['too_close_detected'] += 1
        counters['too_close_detected_sub'].append(f"{tag} | {dist_reason}")
        return False, counters

    # ── Distance-range check ─────────────────────────────────
    # Original validation: all three cathode-to-anode distances must be
    # within [DIST_MIN_MM, DIST_MAX_MM]. See the constants above for why the
    # upper bound is not the 70 mm of the original implementation.
    centers = [c['center_of_mass'] for c in clusters.values()]
    range_ok = False
    accepted_dists = None    # spoke lengths of the permutation that passed
    best_perm_dists = None   # track the closest-to-valid permutation for logging
    best_perm_score = np.inf

    for perm in permutations(centers):
        distances = [euclidean(perm[0], s) for s in perm[1:]]
        if all(DIST_MIN_MM <= d <= DIST_MAX_MM for d in distances):
            range_ok = True
            accepted_dists = list(distances)
            break
        # Track the permutation whose worst violation is smallest
        _mid = (DIST_MIN_MM + DIST_MAX_MM) / 2
        score = max(abs(d - _mid) for d in distances)
        if score < best_perm_score:
            best_perm_score = score
            best_perm_dists = distances

    if not range_ok:
        # Summarise why: which distances were out of range
        out_low  = [f"{d:.1f}" for d in best_perm_dists if d < DIST_MIN_MM]
        out_high = [f"{d:.1f}" for d in best_perm_dists if d > DIST_MAX_MM]
        detail_parts = []
        if out_low:
            detail_parts.append(
                f"too short (<{DIST_MIN_MM:g} mm): {', '.join(out_low)} mm")
        if out_high:
            detail_parts.append(
                f"too long (>{DIST_MAX_MM:g} mm): {', '.join(out_high)} mm")
        detail = "; ".join(detail_parts) if detail_parts else "unknown"
        counters['distance_range_failed'] += 1
        counters['distance_range_failed_sub'].append(
            f"{tag} | 4 clusters, distance-range violation -- {detail}"
        )
        return False, counters

    # ── Planarity check ──────────────────────────────────────
    # Reject if any centre deviates more than PLANARITY_THRESHOLD_MM from
    # the best-fit plane — catches stacked/off-plane segmentation artefacts.
    plane_ok, plane_reason = check_planarity(clusters)
    if not plane_ok:
        counters['planarity_failed'] += 1
        counters['planarity_failed_sub'].append(f"{tag} | {plane_reason}")
        return False, counters

    # ── Star-geometry check ──────────────────────────────────
    # Always measured and recorded; only enforced when a threshold is set.
    hub, radius_cv, min_angle, angle_std = star_metrics(centers)
    counters['star_metrics'][tag] = (radius_cv, min_angle, angle_std, hub)

    # Spoke lengths measured FROM THE STAR HUB, not from whichever permutation
    # satisfied the distance-range check. Those are not the same thing: with
    # `permutations` the first arrangement that fits is accepted, and for an
    # equilateral star the cathode-to-cathode distance is spoke*sqrt(3), which
    # falls inside a wide window. A cathode can therefore pass as the hub, and
    # the audit would then report a cathode-cathode distance as a spoke. The
    # hub from star_metrics is the anatomically meaningful centre.
    _hubc = np.array(centers[hub], dtype=float)
    _spokes = sorted(float(np.linalg.norm(np.array(centers[j], dtype=float) - _hubc))
                     for j in range(4) if j != hub)
    counters['accepted_spokes'][tag] = tuple(_spokes)

    reasons = []
    if STAR_MIN_ANGLE_DEG is not None and min_angle < STAR_MIN_ANGLE_DEG:
        reasons.append(f"min inter-spoke angle {min_angle:.1f} deg "
                       f"< {STAR_MIN_ANGLE_DEG:.1f}")
    if STAR_MAX_RADIUS_CV is not None and radius_cv > STAR_MAX_RADIUS_CV:
        reasons.append(f"spoke-length CV {radius_cv:.3f} "
                       f"> {STAR_MAX_RADIUS_CV:.3f}")
    if reasons:
        counters['star_failed'] += 1
        counters['star_failed_sub'].append(f"{tag} | " + "; ".join(reasons))
        return False, counters

    return True, counters


# ============================================================
#  MAIN PROCESSING
# ============================================================

def process_nifti_files(base_path, log: Logger, network_key: str = None):

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
        # Five-cluster recovery tracking
        'five_cluster_recovery_success':    0,
        'five_cluster_recovery_failed':     0,
        # Geometry-check failures (4 clusters present but invalid geometry)
        'too_close_detected':               0,
        'distance_range_failed':            0,
        'planarity_failed':                 0,
        # Fragment-merge recovery tracking
        'merge_recovery_success':           0,
        'merge_recovery_failed':            0,
        'three_mask_detected_sub':          [],
        'two_mask_detected_sub':            [],
        'one_mask_detected_sub':            [],
        'no_mask_detected_sub':             [],
        'more_then_four_mask_detected_sub': [],
        # Subjects where 5 clusters were found and a spurious one was removed
        'five_cluster_recovery_success_sub': [],
        'five_cluster_recovery_failed_sub':  [],
        # Geometry-check failure subjects (include per-case reason in the tag)
        'too_close_detected_sub':           [],
        'distance_range_failed_sub':        [],
        'planarity_failed_sub':             [],
        # Fragment-merge recovery subjects
        'merge_recovery_success_sub':       [],
        'merge_recovery_failed_sub':        [],
        # 3-cluster split recovery (anode/cathode fused at the edge)
        'split_recovery_success':           0,
        'split_recovery_failed':            0,
        'split_recovery_success_sub':       [],
        'split_recovery_failed_sub':        [],

        # ── Terminal outcome, ONE entry per image ──────────────────────
        # The counters above are EVENT counters: a single image can raise
        # several of them (detected as 3 clusters -> split recovery attempted
        # -> split failed) and summing them therefore double-counts. These two
        # structures record the single terminal outcome per image, so the
        # breakdown below is guaranteed to partition the invalid images
        # exactly once each.
        # tag -> (min, mid, max) spoke length of the accepted configuration
        'accepted_spokes':                  {},
        # tag -> (anode, c1, c2, c3) electrode volumes in mm^3
        'electrode_volumes':                {},

        'star_failed':                      0,
        'star_failed_sub':                  [],
        # tag -> (radius_cv, min_angle, angle_std, hub) for every image that
        # reached the star check, whether or not it passed.
        'star_metrics':                     {},

        'terminal_reason':                  {},   # tag -> reason string
        'valid_tags':                       set(),
    }

    bar = ProgressBar(total=len(nifti_files), label='Extracting coordinates', width=35)

    for file_path in nifti_files:
        counters['total_images'] += 1

        # ── Parse subject / session / run from filename ─────
        # Pattern: rsub-001_ses-2_acq-petra_run-01_PDw_inference.nii.gz
        # The leading 'r' before sub- is stripped before parsing.
        fname  = os.path.basename(file_path)
        clean  = re.sub(r'^r(?=sub-)', '', fname)
        sub_m  = re.search(r'sub-([^_/]+)',  clean)
        ses_m  = re.search(r'ses-([^_/]+)',  clean)
        run_m  = re.search(r'run-([^_/]+)',  clean)
        parts  = file_path.split(os.sep)

        # Fallback: when the filename carries no BIDS tokens (e.g. the file is
        # simply 'petra_inference.nii.gz'), recover subject/session/run by
        # scanning the *whole path* for the corresponding directory token,
        # rather than assuming a fixed depth. This is robust to the
        # electrode_extraction/ level sitting between sub-* and ses-*.
        def _from_path(prefix):
            for p in reversed(parts):
                m = re.fullmatch(rf'{prefix}-([^_/]+)', p)
                if m:
                    return f'{prefix}-{m.group(1)}'
            return None

        subject = f'sub-{sub_m.group(1)}' if sub_m else (_from_path('sub') or 'sub-UNKNOWN')
        session = f'ses-{ses_m.group(1)}' if ses_m else (_from_path('ses') or 'ses-UNKNOWN')
        run     = f'run-{run_m.group(1)}' if run_m else (_from_path('run') or 'run-UNKNOWN')
        tag     = f'{subject}_{session}_{run}'

        try:
            # ── Load NIfTI ───────────────────────────────────
            nii_img  = nib.load(file_path)
            affine   = nii_img.affine
            img_data = nii_img.get_fdata()

            # ── Cluster detection ────────────────────────────
            clusters = find_electrode_clusters(img_data)

            # ── Electrode-size sanity check (warnings only) ──
            # Warn when any cluster is geometrically implausible
            # (too small or too large for a real electrode disc).
            # These are advisory — processing always continues.
            size_warnings = check_electrode_size(clusters, affine)
            for sw in size_warnings:
                log.warn(
                    f"SIZE WARN  {subject} {session} {run}  —  {sw}",
                    print_also=False,
                )

            # ── Five-cluster recovery ────────────────────────
            # When the size-based filter still leaves 5 clusters, try to
            # identify and drop the spurious one before validation.
            if len(clusters) == 5:
                filtered, removed_key, reason = filter_fifth_cluster(clusters)

                if filtered is not None:
                    # Recovery succeeded — log details and proceed with 4 clusters
                    counters['five_cluster_recovery_success'] += 1
                    counters['five_cluster_recovery_success_sub'].append(tag)
                    log.warn(
                        f"5-CLUSTER RECOVERY  {subject} {session} {run}  —  "
                        f"Originally 5 clusters detected. Spurious cluster removed. "
                        f"{reason}",
                        print_also=False,
                    )
                    clusters = filtered
                else:
                    # Recovery failed — mark as invalid (>4 masks) and continue
                    counters['five_cluster_recovery_failed'] += 1
                    counters['five_cluster_recovery_failed_sub'].append(tag)
                    log.warn(
                        f"5-CLUSTER RECOVERY FAILED  {subject} {session} {run}  —  "
                        f"{reason}",
                        print_also=False,
                    )

            # ── Multi-cluster recovery (n = 5 … MAX_RECOVERY_CLUSTERS) ──
            # If the cluster count is still not 4 after the 5-cluster fast
            # path above, run the general recovery.  This covers every
            # combination of fragment-merging and spurious-cluster removal
            # for n = 5, 6, 7, or 8 clusters:
            #
            #   n=5 : drop 1 noise blob  OR  merge 1 pair
            #   n=6 : drop 2  /  merge 2 pairs  /  merge 1 triplet  /
            #         merge 1 pair + drop 1
            #   n=7 : drop 3  /  merge 3 pairs  /  merge 2 pairs+drop 1  /
            #         merge 1 triplet+merge 1 pair  /  merge 1 triplet+drop 1 /
            #         merge 1 pair+drop 2
            #   n=8 : all of the above extended by one more operation
            #
            # Clusters not assigned to any electrode group are discarded with
            # a size-proportional penalty, steering the optimiser away from
            # silently throwing away large real-electrode fragments.
            if len(clusters) != 4 and len(clusters) >= 5:
                merged, merge_reason = recover_four_electrodes(clusters, affine)
                if merged is not None:
                    counters['merge_recovery_success'] += 1
                    counters['merge_recovery_success_sub'].append(tag)
                    log.warn(
                        f"MERGE RECOVERY  {subject} {session} {run}  —  "
                        f"{merge_reason}",
                        print_also=False,
                    )
                    clusters = merged
                else:
                    counters['merge_recovery_failed'] += 1
                    counters['merge_recovery_failed_sub'].append(tag)
                    log.warn(
                        f"MERGE RECOVERY FAILED  {subject} {session} {run}  —  "
                        f"{merge_reason}",
                        print_also=False,
                    )

            # ── Three-cluster split recovery ─────────────────
            # When exactly 3 clusters remain, the usual cause is two
            # neighbouring electrodes (commonly anode + one cathode) fused at
            # their touching edges into a single oversized, elongated cluster.
            # Attempt to split that cluster back into two electrode-sized discs.
            # The split is accepted only if both halves look electrode-sized and
            # are not closer than the minimum inter-electrode distance, so a
            # genuinely missing electrode is left as an (invalid) 3-cluster set.
            if len(clusters) == 3:
                split_clusters, split_reason = try_split_merged_cluster(clusters, affine)
                if split_clusters is not None:
                    counters['split_recovery_success'] += 1
                    counters['split_recovery_success_sub'].append(tag)
                    log.warn(
                        f"3-CLUSTER SPLIT RECOVERY  {subject} {session} {run}  —  "
                        f"{split_reason}",
                        print_also=False,
                    )
                    clusters = split_clusters
                else:
                    counters['split_recovery_failed'] += 1
                    counters['split_recovery_failed_sub'].append(tag)
                    log.warn(
                        f"3-CLUSTER SPLIT FAILED  {subject} {session} {run}  —  "
                        f"{split_reason}",
                        print_also=False,
                    )

            # ── Validate spatial configuration ───────────────
            is_valid, counters = is_valid_configuration(
                clusters, counters, subject, session, run
            )

            if is_valid:
                counters['valid_configurations'] += 1
                cl = list(clusters.values())

                # ── Identify the anode geometrically ─────────────────────
                # cl[0] is whatever cluster scipy.ndimage.label numbered
                # first — it is NOT necessarily the central electrode.
                # Use _star_score to find which of the 4 clusters is the
                # geometric centre (anode) and order the remaining 3 as
                # cathodes, preserving their original relative order.
                vox_centers = [c['center_of_mass'] for c in cl]
                _, anode_idx = _star_score(vox_centers)
                ordered = (
                    [cl[anode_idx]]
                    + [c for i, c in enumerate(cl) if i != anode_idx]
                )

                _sm = counters['star_metrics'].get(tag, (None, None, None, None))
                _sk = counters['accepted_spokes'].get(tag, (None, None, None))

                # Electrode volumes, in anode-then-cathodes order. All four are
                # the same physical object, so vol_cv near 0 is expected; a
                # large value means one cluster is the wrong size -- typically
                # a fused pair (~2x) or a fragment (~0.5x).
                _vols = _cluster_volumes_mm3(ordered, affine)
                _vmean = sum(_vols) / len(_vols)
                _vcv = (float(np.std(_vols)) / _vmean) if _vmean > 0 else None
                counters['electrode_volumes'][tag] = tuple(_vols)
                results.append({
                    # Carried into the CSV so a merged table can always be
                    # traced back to the network that produced the mask.
                    'network':      network_key or NETWORK_KEY,
                    # Star-geometry quality, recorded for every valid image so
                    # a threshold can be chosen from the real distribution.
                    'vol_anode_mm3':  round(_vols[0], 1),
                    'vol_cathode1_mm3': round(_vols[1], 1),
                    'vol_cathode2_mm3': round(_vols[2], 1),
                    'vol_cathode3_mm3': round(_vols[3], 1),
                    'vol_total_mm3':  round(sum(_vols), 1),
                    'vol_cv':         None if _vcv is None else round(_vcv, 4),
                    'spoke_min_mm':   None if _sk[0] is None else round(_sk[0], 2),
                    'spoke_max_mm':   None if _sk[2] is None else round(_sk[2], 2),
                    'star_radius_cv': None if _sm[0] is None else round(_sm[0], 4),
                    'star_min_angle': None if _sm[1] is None else round(_sm[1], 2),
                    'star_angle_std': None if _sm[2] is None else round(_sm[2], 2),
                    # True when the widest-spread hub agrees with the hub
                    # _star_score picks (the one used to label the anode).
                    'star_hub_agrees': None if _sm[3] is None else bool(_sm[3] == anode_idx),
                    'subject':      subject,
                    'session':      session,
                    'run':          run,
                    'anode':    voxel_to_corr_space(ordered[0]['center_of_mass'], affine),
                    'cathode1': voxel_to_corr_space(ordered[1]['center_of_mass'], affine),
                    'cathode2': voxel_to_corr_space(ordered[2]['center_of_mass'], affine),
                    'cathode3': voxel_to_corr_space(ordered[3]['center_of_mass'], affine),
                })

                counters['valid_tags'].add(tag)
                log.info(
                    f"OK       {subject} {session} {run}  —  "
                    f"{len(clusters)} clusters, coordinates extracted.",
                    print_also=False,
                )
            else:
                counters['invalid_configurations'] += 1
                # Build a more specific reason tag for geometry failures
                if counters['too_close_detected_sub'] and \
                        counters['too_close_detected_sub'][-1].startswith(tag):
                    reason_tag = "too-close pair"
                elif counters['distance_range_failed_sub'] and \
                        counters['distance_range_failed_sub'][-1].startswith(tag):
                    reason_tag = "distance-range violation"
                elif counters['planarity_failed_sub'] and \
                        counters['planarity_failed_sub'][-1].startswith(tag):
                    reason_tag = "planarity failure"
                elif counters['star_failed_sub'] and \
                        counters['star_failed_sub'][-1].startswith(tag):
                    reason_tag = 'star-geometry violation'
                else:
                    reason_tag = f"{len(clusters)} cluster(s)"
                # Exactly one terminal reason per invalid image.
                counters['terminal_reason'][tag] = reason_tag
                log.warn(
                    f"INVALID  {subject} {session} {run}  —  "
                    f"{reason_tag}, configuration rejected.",
                    print_also=False,
                )

        except Exception as exc:
            counters['invalid_configurations'] += 1
            counters['terminal_reason'][tag] = 'exception during processing'
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
        "  ── geometry checks (4 clusters) ───────────────",
        f"  Too-close pair (<15 mm) : {counters['too_close_detected']}",
        f"  Distance range ({DIST_MIN_MM:g}-{DIST_MAX_MM:g} mm): "
        f"{counters['distance_range_failed']}",
        f"  Planarity failure       : {counters['planarity_failed']}",
        "  ── 5-cluster recovery ─────────────────────────",
        f"  Recovered (5→4 fixed)   : {counters['five_cluster_recovery_success']}",
        f"  Recovery failed (5→inv) : {counters['five_cluster_recovery_failed']}",
        "  ── fragment-merge / general recovery ─────────────────",
        f"  Recovered (N→4 fixed)      : {counters['merge_recovery_success']}",
        f"  Recovery failed (N→invalid): {counters['merge_recovery_failed']}",
        "  ── 3-cluster split recovery (fused electrodes) ──",
        f"  Recovered (3→4 fixed)      : {counters['split_recovery_success']}",
        f"  Split failed (left as 3)   : {counters['split_recovery_failed']}",
    ]
    # ── Terminal-reason breakdown ───────────────────────────────────────
    # The counters printed above are EVENT counters and legitimately overlap:
    # an image detected as 3 clusters, sent to split recovery, and rejected
    # raises 'three_mask_detected' AND 'split_recovery_failed'. Summing them
    # double-counts, which is why the old sanity check could report a NEGATIVE
    # number of unaccounted invalids while genuinely unexplained images were
    # hidden inside the overlap. The breakdown below is built from one terminal
    # reason per image, so it partitions the invalid set exactly.
    from collections import Counter as _Counter
    reason_counts = _Counter(counters['terminal_reason'].values())
    n_reasons     = sum(reason_counts.values())
    n_invalid     = counters['invalid_configurations']
    unaccounted   = n_invalid - n_reasons

    # ── Distance audit ──────────────────────────────────────────────────
    # Accepted configurations, tiered by their LARGEST spoke. Everything above
    # DIST_AUDIT_NAME_ABOVE is named so it can be opened and checked: the bound
    # is set at a gap in the data, which says those images are not stray
    # clusters -- it does not say the electrodes are correctly placed.
    counters['_distance_audit_sub'] = [
        f"{tg} | longest spoke {v[-1]:.1f} mm (spokes "
        f"{v[0]:.1f}/{v[1]:.1f}/{v[2]:.1f} mm)"
        for tg, v in sorted(counters['accepted_spokes'].items(),
                            key=lambda kv: -kv[1][-1])
        if v[-1] >= DIST_AUDIT_NAME_ABOVE
    ]
    _sp = counters['accepted_spokes']
    if _sp:
        maxes = {tag: v[-1] for tag, v in _sp.items()}
        summary += ["  ── distance audit (accepted images, longest spoke) ──"]
        prev = DIST_MIN_MM
        for tier in DIST_AUDIT_TIERS:
            n = sum(1 for m in maxes.values() if prev <= m < tier)
            if n:
                summary.append(f"  {prev:>5.0f} - {tier:<5.0f} mm      : {n}")
            prev = tier
        n_top = sum(1 for m in maxes.values() if m >= DIST_AUDIT_TIERS[-1])
        summary.append(
            f"  >= {DIST_AUDIT_TIERS[-1]:.0f} mm           : {n_top}")

        flagged = sorted(((m, tg) for tg, m in maxes.items()
                          if m >= DIST_AUDIT_NAME_ABOVE), reverse=True)
        summary.append(
            f"  above {DIST_AUDIT_NAME_ABOVE:.0f} mm         : {len(flagged)} "
            f"of {len(maxes)} accepted "
            f"({len(flagged)/len(maxes)*100:.2f}%)")
        if flagged:
            summary.append("  ── CHECK THESE (longest spoke, mm) ────────────")
            for m, tg in flagged[:40]:
                summary.append(f"    {m:7.1f}   {tg}")
            if len(flagged) > 40:
                summary.append(f"    ... and {len(flagged) - 40} more "
                               f"(all listed in the detail section below)")

    # ── Electrode-volume distribution ───────────────────────────────────
    # The four electrodes are identical physical objects. Their segmented
    # volumes are therefore an ABSOLUTE reference that needs no ground truth:
    # compare the median against the known electrode size to detect systematic
    # over- or under-segmentation, and compare vol_cv across networks to see
    # which one separates adjacent electrodes most cleanly.
    _ev = counters['electrode_volumes']
    if _ev:
        per_electrode = sorted(v for vols in _ev.values() for v in vols)
        cvs = sorted(float(np.std(v)) / (sum(v) / len(v))
                     for v in _ev.values() if sum(v) > 0)
        def _q(xs, q):
            return xs[min(len(xs) - 1, int(q / 100 * len(xs)))]
        summary += [
            "  ── electrode volume (mm3, valid images) ───────",
            f"  electrodes measured     : {len(per_electrode)}",
            f"  per-electrode volume         p5={_q(per_electrode,5):.0f}  "
            f"p50={_q(per_electrode,50):.0f}  p95={_q(per_electrode,95):.0f}",
            f"  within-image volume CV       p50={_q(cvs,50):.3f}  "
            f"p95={_q(cvs,95):.3f}  max={cvs[-1]:.3f}",
        ]

    # ── Star-geometry distribution ──────────────────────────────────────
    # Printed for every image that reached the check. Use these percentiles to
    # set STAR_MIN_ANGLE_DEG / STAR_MAX_RADIUS_CV -- do not guess them.
    _sm = counters['star_metrics']
    if _sm:
        cvs  = sorted(v[0] for v in _sm.values())
        angs = sorted(v[1] for v in _sm.values())
        disagree = sum(1 for v in _sm.values() if v[3] is not None)
        def _pct(xs, q):
            return xs[min(len(xs) - 1, int(q / 100 * len(xs)))]
        summary += [
            "  ── star geometry (measured on all 4-cluster images) ──",
            f"  images measured         : {len(_sm)}",
            "  min inter-spoke angle (deg)  "
            f"p1={_pct(angs,1):.1f}  p5={_pct(angs,5):.1f}  "
            f"p50={_pct(angs,50):.1f}  p95={_pct(angs,95):.1f}",
            "  spoke-length CV              "
            f"p50={_pct(cvs,50):.3f}  p95={_pct(cvs,95):.3f}  "
            f"p99={_pct(cvs,99):.3f}  max={cvs[-1]:.3f}",
            f"  rejected by star check  : {counters['star_failed']}"
            + ("   (thresholds not set -- measure only)"
               if STAR_MIN_ANGLE_DEG is None and STAR_MAX_RADIUS_CV is None else ""),
        ]

    summary += [
        "  ── terminal reason (one per invalid image) ────",
    ]
    for reason, n in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        summary.append(f"  {reason:<24}: {n}")
    summary += [
        "  ── sanity check ───────────────────────────────",
        f"  Invalid configurations  : {n_invalid}",
        f"  Terminal reasons logged : {n_reasons}",
        f"  Unaccounted invalids    : {unaccounted}"
        + ("  ok" if unaccounted == 0 else "  <- an invalid path sets no reason"),
    ]

    # Cross-check the event counters against the terminal reasons. A large gap
    # is not an error -- recovery attempts are events, not outcomes -- but it
    # is worth showing so the two kinds of number are never confused.
    events = (
        counters['no_mask_detected'] + counters['one_mask_detected'] +
        counters['two_mask_detected'] + counters['three_mask_detected'] +
        counters['more_then_four_mask_detected'] +
        counters['too_close_detected'] + counters['distance_range_failed'] +
        counters['planarity_failed'] +
        counters['five_cluster_recovery_failed'] +
        counters['merge_recovery_failed'] +
        counters['split_recovery_failed']
    )
    summary += [
        f"  Event counters (overlap): {events}"
        f"   [{events - n_invalid:+d} vs invalid; overlap is expected]",
    ]

    # Valid + invalid must equal the number of images seen. This is the check
    # that actually guards the headline extraction-success rate.
    seen = len(counters['valid_tags']) + len(counters['terminal_reason'])
    summary += [
        f"  Valid + invalid images  : {seen} of {counters['total_images']}"
        + ("  ok" if seen == counters['total_images']
           else "  <- images with no recorded outcome"),
        "=" * 55,
    ]
    for line in summary:
        print(line)
        log._fh.write(line + '\n')
    log._fh.flush()          # summary on disk even if the run is killed later

    # Log every affected subject per invalid category
    invalid_categories = [
        ('no_mask_detected_sub',              'No mask'),
        ('one_mask_detected_sub',             'One mask'),
        ('two_mask_detected_sub',             'Two masks'),
        ('three_mask_detected_sub',           'Three masks'),
        ('more_then_four_mask_detected_sub',  'More than four masks'),
        ('too_close_detected_sub',            '4 clusters but too-close pair (< 15 mm)'),
        ('_distance_audit_sub',
         f'ACCEPTED but longest spoke >= {DIST_AUDIT_NAME_ABOVE:g} mm '
         f'-- verify these visually'),
        ('distance_range_failed_sub',
         f'4 clusters but distance-range violation '
         f'({DIST_MIN_MM:g}-{DIST_MAX_MM:g} mm)'),
        ('planarity_failed_sub',              '4 clusters but planarity check failed'),
        ('five_cluster_recovery_success_sub', '5-cluster: spurious electrode removed successfully'),
        ('five_cluster_recovery_failed_sub',  '5-cluster: recovery failed, marked invalid'),
        ('merge_recovery_success_sub',        'Fragment-merge: clusters merged to valid 4-electrode config'),
        ('merge_recovery_failed_sub',         'Fragment-merge: merge failed, marked invalid'),
        ('split_recovery_success_sub',        '3-cluster: fused electrodes split successfully'),
        ('split_recovery_failed_sub',         '3-cluster: split failed, left as 3'),
        ('star_failed_sub',                   '4 clusters but star-geometry check failed'),
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

def _parse_args():
    ap = argparse.ArgumentParser(
        description="Extract electrode coordinates from one network's "
                    "segmentation masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--network", default=DEFAULT_NETWORK,
        help="Which network's inference files to process. This is the suffix "
             "written by Segmentation_paper_models.py, i.e. it selects "
             "*_inference_<network>.nii.gz. Also used to name the output CSV "
             "and the log file. Use --list to see the known keys.",
    )
    ap.add_argument(
        "--base-path", dest="base_path",
        default='/media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction',
        help="Root folder containing sub-*/unzipped/ .",
    )
    ap.add_argument(
        "--out-dir", dest="out_dir", default=None,
        help="Folder for the output CSV (default: Tables/ next to this script).",
    )
    ap.add_argument(
        "--out-name", dest="out_name", default=None,
        help="Override the output CSV filename entirely.",
    )
    ap.add_argument(
        "--label", default=None,
        help="Extra label inserted into the output filename, e.g. a cohort "
             "name. Default: none.",
    )
    ap.add_argument(
        "--dist-max", dest="dist_max", type=float, default=None,
        help="Upper bound on the cathode-to-anode distance (mm). Defaults to "
             f"the module constant. Use --dist-max 70 to reproduce the "
             "figures published under the original bound.",
    )
    ap.add_argument(
        "--audit-above", dest="audit_above", type=float, default=None,
        help="Name every accepted image whose longest spoke exceeds this "
             "(mm) in the summary, so it can be opened and checked. "
             f"Default: {DIST_AUDIT_NAME_ABOVE:g}.",
    )
    ap.add_argument(
        "--dist-min", dest="dist_min", type=float, default=None,
        help="Lower bound on the cathode-to-anode distance (mm).",
    )
    ap.add_argument(
        "--star-min-angle", dest="star_min_angle", type=float, default=None,
        help="Reject configurations whose smallest inter-spoke angle is below "
             "this (degrees). Omit to measure without rejecting. Set it from "
             "the percentiles printed in the summary, not from a guess.",
    )
    ap.add_argument(
        "--star-max-radius-cv", dest="star_max_radius_cv", type=float,
        default=None,
        help="Reject configurations whose spoke-length coefficient of "
             "variation exceeds this. Omit to measure without rejecting.",
    )
    ap.add_argument(
        "--list", action="store_true",
        help="List the known network keys and how many inference files exist "
             "for each under --base-path, then exit.",
    )
    return ap.parse_args()


if __name__ == "__main__":

    args = _parse_args()

    # ── Inventory mode ───────────────────────────────────────
    if args.list:
        print(f"\n  Base path: {args.base_path}\n")
        print(f"  {'key':<20}{'files':>7}   description")
        print(f"  {'-'*20}{'-'*7}   {'-'*40}")
        for key, desc in NETWORKS.items():
            n = len(find_nifti_files(args.base_path, key))
            print(f"  {key:<20}{n:>7}   {desc}")
        # anything on disk that is not in the registry
        seen = set()
        for p in glob.glob(os.path.join(args.base_path, "sub-*", "unzipped",
                                        "*_inference_*.nii.gz")):
            m = re.search(r"_inference_(.+)\.nii\.gz$", os.path.basename(p))
            if m and m.group(1) not in NETWORKS:
                seen.add(m.group(1))
        if seen:
            print("\n  Suffixes on disk that are NOT in the registry:")
            for s in sorted(seen):
                print(f"    {s}   (add it to NETWORKS, or pass --network {s})")
        print()
        raise SystemExit(0)

    # ── Resolve configuration ────────────────────────────────
    NETWORK_KEY = args.network
    if args.audit_above is not None:
        DIST_AUDIT_NAME_ABOVE = args.audit_above
    if args.dist_min is not None:
        DIST_MIN_MM = args.dist_min
    if args.dist_max is not None:
        DIST_MAX_MM = args.dist_max
    STAR_MIN_ANGLE_DEG = args.star_min_angle
    STAR_MAX_RADIUS_CV = args.star_max_radius_cv
    if NETWORK_KEY not in NETWORKS:
        print(f"\n  NOTE: '{NETWORK_KEY}' is not in the NETWORKS registry. "
              f"Proceeding anyway — the glob is *_inference_{NETWORK_KEY}.nii.gz.\n")

    base_path  = args.base_path
    Table_path = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / 'Tables'

    stamp = datetime.now().strftime('%Y%m%d')
    if args.out_name:
        out_name = args.out_name
    else:
        # NOTE: do not name this `label` -- assignments in the __main__ block
        # are module-scope, and `label` is scipy.ndimage.label, imported at the
        # top and used by find_electrode_clusters(). Shadowing it makes every
        # image fail with "'str' object is not callable".
        label_part = f"_{args.label}" if args.label else ""
        # Network key first, so `ls Tables/` groups by network.
        out_name = f"electrode_positions_{NETWORK_KEY}{label_part}_{stamp}.csv"
    output_csv = os.path.join(Table_path, out_name)

    # Log file carries the network too, so parallel runs of different
    # networks cannot overwrite each other's log.
    LOG_FILE = TMP_DIR / (
        f"log_extract_coordinates_{NETWORK_KEY}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    if os.path.exists(output_csv):
        print(f"\n  WARNING: {output_csv} already exists and will be "
              f"overwritten.\n")

    # ── Start logger ─────────────────────────────────────────
    log = Logger(LOG_FILE)
    log.info("Script started")
    log.info(f"Network    : {NETWORK_KEY}  ({NETWORKS.get(NETWORK_KEY, 'not in registry')})")
    log.info(f"Mask glob  : *_inference_{NETWORK_KEY}.nii.gz")
    log.info(f"Base path  : {base_path}")
    log.info(f"Output CSV : {output_csv}")
    log.info(f"Log file   : {LOG_FILE}\n")

    # The same header goes to the log, so a saved log is self-contained: the
    # thresholds a run used are as important as its results.
    log.info(f"Distance window : {DIST_MIN_MM:g}-{DIST_MAX_MM:g} mm")
    log.info(f"Star min angle  : "
             f"{'not enforced' if STAR_MIN_ANGLE_DEG is None else f'{STAR_MIN_ANGLE_DEG:g} deg'}")
    log.info(f"Star max rad CV : "
             f"{'not enforced' if STAR_MAX_RADIUS_CV is None else f'{STAR_MAX_RADIUS_CV:g}'}")
    log.info(f"Audit names above: {DIST_AUDIT_NAME_ABOVE:g} mm\n")

    print(f"\n{'='*62}")
    print(f"  Extract Electrode Coordinates")
    print(f"{'='*62}")
    print(f"  Network    : {NETWORK_KEY}")
    print(f"               {NETWORKS.get(NETWORK_KEY, '(not in registry)')}")
    print(f"  Mask glob  : *_inference_{NETWORK_KEY}.nii.gz")
    print(f"  Base path  : {base_path}")
    print(f"  Output CSV : {output_csv}")
    print(f"  Log file   : {LOG_FILE}")
    print(f"{'='*62}\n")

    # ── Run ──────────────────────────────────────────────────
    df = process_nifti_files(base_path, log, network_key=NETWORK_KEY)

    # ── Save results ─────────────────────────────────────────
    if not df.empty:
        os.makedirs(Table_path, exist_ok=True)
        df.to_csv(output_csv, index=False)
        msg = f"Results saved -> {output_csv}  ({len(df)} rows)"
        print(f"\n  {msg}")
        log.info(msg)
    else:
        msg = ("No valid results to save - CSV not written. "
               f"Check that *_inference_{NETWORK_KEY}.nii.gz files exist "
               f"under {base_path} (try --list).")
        print(f"\n  {msg}")
        log.warn(msg)

    print(f"  Log file   : {LOG_FILE}\n")
    log.close()
