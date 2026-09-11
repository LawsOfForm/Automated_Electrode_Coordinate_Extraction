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


def _is_valid_star(centers: list,
                   dist_min: float = 5.0,
                   dist_max: float = 70.0) -> bool:
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
                          dist_min: float = 5.0,
                          dist_max: float = 70.0) -> tuple[dict | None, int | None, str]:
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

def find_nifti_files(base_path):
    pattern = os.path.join(base_path, "sub-*", "unzipped", "*inference.nii.gz")
    # Method comparison study:
    # pattern = os.path.join(base_path, "sub-*", "electrode_extraction",
    #                        "ses*", "run*", "petra_inference.nii.gz")
    return glob.glob(pattern)


def voxel_to_corr_space(voxel_coords, affine):
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
    # between 5 mm and 70 mm (30–70 mm in practice for this montage).
    centers = [c['center_of_mass'] for c in clusters.values()]
    range_ok = False
    best_perm_dists = None   # track the closest-to-valid permutation for logging
    best_perm_score = np.inf

    for perm in permutations(centers):
        distances = [euclidean(perm[0], s) for s in perm[1:]]
        if all(5 <= d <= 70 for d in distances):
            range_ok = True
            break
        # Track the permutation whose worst violation is smallest
        score = max(abs(d - 37.5) for d in distances)   # 37.5 = midpoint of 5-70
        if score < best_perm_score:
            best_perm_score = score
            best_perm_dists = distances

    if not range_ok:
        # Summarise why: which distances were out of range
        out_low  = [f"{d:.1f}" for d in best_perm_dists if d <  5]
        out_high = [f"{d:.1f}" for d in best_perm_dists if d > 70]
        detail_parts = []
        if out_low:
            detail_parts.append(f"too short (<5 mm): {', '.join(out_low)} mm")
        if out_high:
            detail_parts.append(f"too long (>70 mm): {', '.join(out_high)} mm")
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

    return True, counters


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
    }

    bar = ProgressBar(total=len(nifti_files), label='Extracting coordinates', width=35)

    for file_path in nifti_files:
        counters['total_images'] += 1

        # ── Parse subject / session / run from filename ─────
        # Pattern: rsub-001_ses-2_acq-petra_run-01_PDw_inference.nii.gz
        # The leading 'r' before sub- is stripped before parsing.
        fname  = os.path.basename(file_path)
        clean  = re.sub(r'^r(?=sub-)', '', fname)
        sub_m  = re.search(r'sub-([^_]+)',  clean)
        ses_m  = re.search(r'ses-([^_]+)',  clean)
        run_m  = re.search(r'run-([^_]+)',  clean)
        parts  = file_path.split(os.sep)
        subject = f'sub-{sub_m.group(1)}' if sub_m else parts[-4]
        session = f'ses-{ses_m.group(1)}' if ses_m else parts[-3]
        run     = f'run-{run_m.group(1)}' if run_m else parts[-2]
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

                results.append({
                    'subject':      subject,
                    'session':      session,
                    'run':          run,
                    'anode':    voxel_to_corr_space(ordered[0]['center_of_mass'], affine),
                    'cathode1': voxel_to_corr_space(ordered[1]['center_of_mass'], affine),
                    'cathode2': voxel_to_corr_space(ordered[2]['center_of_mass'], affine),
                    'cathode3': voxel_to_corr_space(ordered[3]['center_of_mass'], affine),
                })

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
                else:
                    reason_tag = f"{len(clusters)} cluster(s)"
                log.warn(
                    f"INVALID  {subject} {session} {run}  —  "
                    f"{reason_tag}, configuration rejected.",
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
        "  ── geometry checks (4 clusters) ───────────────",
        f"  Too-close pair (<15 mm) : {counters['too_close_detected']}",
        f"  Distance range (5-70mm) : {counters['distance_range_failed']}",
        f"  Planarity failure       : {counters['planarity_failed']}",
        "  ── 5-cluster recovery ─────────────────────────",
        f"  Recovered (5→4 fixed)   : {counters['five_cluster_recovery_success']}",
        f"  Recovery failed (5→inv) : {counters['five_cluster_recovery_failed']}",
        "  ── fragment-merge / general recovery ─────────────────",
        f"  Recovered (N→4 fixed)      : {counters['merge_recovery_success']}",
        f"  Recovery failed (N→invalid): {counters['merge_recovery_failed']}",
        "  ── sanity check ───────────────────────────────",
    ]
    accounted = (
        counters['no_mask_detected'] + counters['one_mask_detected'] +
        counters['two_mask_detected'] + counters['three_mask_detected'] +
        counters['more_then_four_mask_detected'] +
        counters['too_close_detected'] + counters['distance_range_failed'] +
        counters['planarity_failed'] +
        counters['five_cluster_recovery_failed'] +
        counters['merge_recovery_failed']
    )
    unaccounted = counters['invalid_configurations'] - accounted
    summary += [
        f"  Invalids accounted for  : {accounted}",
        f"  Unaccounted invalids    : {unaccounted}"
        + ("  ✓" if unaccounted == 0 else "  ← check logic!"),
        "=" * 55,
    ]
    for line in summary:
        print(line)
        log._fh.write(line + '\n')

    # Log every affected subject per invalid category
    invalid_categories = [
        ('no_mask_detected_sub',              'No mask'),
        ('one_mask_detected_sub',             'One mask'),
        ('two_mask_detected_sub',             'Two masks'),
        ('three_mask_detected_sub',           'Three masks'),
        ('more_then_four_mask_detected_sub',  'More than four masks'),
        ('too_close_detected_sub',            '4 clusters but too-close pair (< 15 mm)'),
        ('distance_range_failed_sub',         '4 clusters but distance-range violation (5-70 mm)'),
        ('planarity_failed_sub',              '4 clusters but planarity check failed'),
        ('five_cluster_recovery_success_sub', '5-cluster: spurious electrode removed successfully'),
        ('five_cluster_recovery_failed_sub',  '5-cluster: recovery failed, marked invalid'),
        ('merge_recovery_success_sub',        'Fragment-merge: clusters merged to valid 4-electrode config'),
        ('merge_recovery_failed_sub',         'Fragment-merge: merge failed, marked invalid'),
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
