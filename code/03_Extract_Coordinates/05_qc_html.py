#!/usr/bin/env python3
"""
==============================================================
  05_qc_html.py
==============================================================
  Visual quality control for electrode segmentation.

  Writes THREE cross-linked reports per project:

    1_accepted   images from which four coordinates were
                 extracted on the first pass
    2_errors     images the pipeline rejected
    3_rescued    images that FAILED the plausibility checks and
                 were then re-admitted by post-processing --
                 the 5-cluster, fragment-merge and 3-cluster
                 split recovery paths

  The rescued page is the one to review first. Those images were
  rejected on geometry and then let back in by a heuristic, so
  they carry the highest risk in both directions: an incorrect
  extraction admitted, or a correct rejection overturned.

  EACH HEAD GETS FOUR ROWS, one per electrode, each showing the
  sagittal, coronal and axial slice centred and zoomed on that
  electrode. A single mid-brain slice shows whether the
  segmentation is roughly in the right place; it cannot show
  whether an individual electrode is correctly delineated, and
  it is the individual electrode that determines the coordinate.

  The four rows are rendered as ONE image per head rather than
  twelve. Twelve separate JPEGs per head would roughly quadruple
  the page size for identical content.

  Electrodes are identified from the mask itself rather than
  from the coordinate table, so the same code works on rejected
  images where no coordinates exist. With exactly four clusters
  the central one is found by the same widest-spread rule the
  extraction uses; otherwise clusters are numbered by size.

--------------------------------------------------------------
  HOW TO USE
      python 05_qc_html.py --network b_no_attention
      python 05_qc_html.py --network b_no_attention --limit 40
      python 05_qc_html.py --network b_no_attention --single-page
      python 05_qc_html.py --network b_no_attention --zoom 45
==============================================================
"""

import os
import re
import csv
import glob
import base64
import argparse
from io import BytesIO
from itertools import combinations
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib
from scipy.ndimage import label, center_of_mass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_IMAGES = '/media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction'
PLANES = [("Sagittal", 0), ("Coronal", 1), ("Axial", 2)]

# Headings under which 01_Extract_coordinate_ablation_final.py lists the images
# that post-processing recovered. Matching them is what separates page 3.
RECOVERY_HEADINGS = [
    "5-cluster: spurious electrode removed successfully",
    "Fragment-merge: clusters merged to valid 4-electrode config",
    "3-cluster: fused electrodes split successfully",
]
TAG_RE = re.compile(r"^(sub-\w+_ses-\w+_run-\w+)")


def find_inference(images_root, network):
    return sorted(glob.glob(os.path.join(
        images_root, "sub-*", "unzipped", f"*_inference_{network}.nii.gz")))


def tag_from_inference(path, network):
    b = os.path.basename(path).replace(f"_PDw_inference_{network}.nii.gz", "")
    sub = re.search(r"r?(sub-\w+?)_", b)
    ses = re.search(r"(ses-\w+?)_", b)
    run = re.search(r"(run-\w+)", b)
    return "_".join(x.group(1) for x in (sub, ses, run) if x)


def project_of(tag):
    m = re.search(r"sub-(\d)", tag)
    return f"P{m.group(1)}" if m else "unknown"


def load_extraction(csv_path):
    out = {}
    if not csv_path or not os.path.isfile(csv_path):
        return out
    with open(csv_path, encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            out["_".join(r.get(k, "") for k in
                         ("subject", "session", "run"))] = r
    return out


def load_rescued(log_path):
    """Tags listed beneath a recovery heading in the extraction log."""
    rescued, heading = {}, None
    if not log_path or not os.path.isfile(log_path):
        return rescued
    for line in open(log_path, encoding="utf-8", errors="replace"):
        s = line.rstrip()
        if not s.strip():
            continue
        hit = next((h for h in RECOVERY_HEADINGS if h in s), None)
        if hit:
            heading = hit
            continue
        if s.strip().endswith(":") and not any(h in s for h in RECOVERY_HEADINGS):
            heading = None          # a different detail section started
            continue
        if heading:
            m = TAG_RE.match(s.strip())
            if m:
                rescued[m.group(1)] = heading
    return rescued


def fnum(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def clusters_of(mask, min_voxels=20):
    """Cluster voxel counts and centroids, largest first."""
    lab, n = label(mask)
    out = []
    for i in range(1, n + 1):
        sel = lab == i
        v = int(sel.sum())
        if v >= min_voxels:
            out.append((v, tuple(center_of_mass(sel))))
    out.sort(key=lambda t: -t[0])
    return out


def hub_index(centres):
    """Central electrode by the widest-spread rule, as the extraction uses."""
    if len(centres) != 4:
        return None
    best, bi = -1.0, 0
    for h in range(4):
        c = np.array(centres[h], float)
        sp = [np.array(centres[j], float) - c for j in range(4) if j != h]
        ang = []
        for i, j in combinations(range(3), 2):
            ca = np.dot(sp[i], sp[j]) / (np.linalg.norm(sp[i]) *
                                         np.linalg.norm(sp[j]) + 1e-9)
            ang.append(np.degrees(np.arccos(np.clip(ca, -1, 1))))
        if min(ang) > best:
            best, bi = min(ang), h
    return bi


def head_montage(anat_path, mask_path, zoom_mm=60, width_px=640, dpi=100,
                 max_rows=4):
    """One image per head: a row per electrode, three views per row."""
    aimg = nib.load(anat_path)
    anat = np.asanyarray(aimg.dataobj).astype(np.float32)
    mask = np.asanyarray(nib.load(mask_path).dataobj) > 0
    zooms = [float(z) for z in aimg.header.get_zooms()[:3]]

    cl = clusters_of(mask)
    hub = hub_index([c[1] for c in cl[:4]]) if len(cl) == 4 else None
    if hub is not None:
        cl = [cl[hub]] + [cl[i] for i in range(4) if i != hub]
        names = ["Anode (centre)", "Cathode 1", "Cathode 2", "Cathode 3"]
    else:
        names = [f"Cluster {i + 1}" for i in range(max(len(cl), 1))]

    rows = cl[:max_rows] if cl else [(0, tuple(s // 2 for s in anat.shape))]
    if not cl:
        names = ["no mask detected"]

    lo, hi = np.percentile(anat, [1, 99.5])
    anat = np.clip((anat - lo) / max(hi - lo, 1e-6), 0, 1)

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, dpi=dpi, squeeze=False,
                             figsize=(width_px / dpi,
                                      width_px / 3 * n_rows / dpi))
    for r, (vox, centre) in enumerate(rows):
        c = [int(round(v)) for v in centre]
        for k, (pname, axis) in enumerate(PLANES):
            ax = axes[r][k]
            idx = min(max(c[axis], 0), anat.shape[axis] - 1)
            a = np.take(anat, idx, axis=axis).T
            m = np.take(mask, idx, axis=axis).T
            ax.imshow(a, cmap="gray", origin="lower", interpolation="nearest")
            if m.any():
                ax.contour(m, levels=[0.5], colors="#ff2d2d", linewidths=0.8)
                ax.imshow(np.ma.masked_where(~m, m), cmap="autumn", alpha=0.30,
                          origin="lower", interpolation="nearest")
            # Zoom on this electrode. The point of a per-electrode row is to
            # see the delineation, which a whole-head slice cannot show.
            oth = [i for i in range(3) if i != axis]
            ax.set_xlim(c[oth[0]] - zoom_mm / 2 / zooms[oth[0]],
                        c[oth[0]] + zoom_mm / 2 / zooms[oth[0]])
            ax.set_ylim(c[oth[1]] - zoom_mm / 2 / zooms[oth[1]],
                        c[oth[1]] + zoom_mm / 2 / zooms[oth[1]])
            if r == 0:
                ax.set_title(pname, fontsize=9, color="w", pad=4)
            if k == 0:
                # Backing boxes: the labels sit over bright scalp and anatomy,
                # where unbacked text is unreadable at this size.
                bb = dict(boxstyle="round,pad=0.25", facecolor="#000000cc",
                          edgecolor="none")
                ax.text(0.03, 0.96, names[r] if r < len(names) else "",
                        transform=ax.transAxes, fontsize=8.5, color="#ffe08a",
                        va="top", ha="left", bbox=bb, zorder=6)
                if vox:
                    ax.text(0.03, 0.04, f"{vox} vox", transform=ax.transAxes,
                            fontsize=7.5, color="#9fd8ff", va="bottom",
                            ha="left", bbox=bb, zorder=6)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#333333")
    fig.subplots_adjust(0, 0, 1, 0.97, 0.012, 0.012)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02,
                facecolor="black")
    plt.close(fig)
    raw = buf.getvalue()
    try:
        from PIL import Image
        im = Image.open(BytesIO(raw)).convert("RGB")
        out = BytesIO()
        im.save(out, format="JPEG", quality=68, optimize=True)
        return "jpeg", base64.b64encode(out.getvalue()).decode("ascii")
    except Exception:                                           # noqa: BLE001
        return "png", base64.b64encode(raw).decode("ascii")


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;
     background:#11151a;color:#e8ecf1}
header{position:sticky;top:0;background:#171c23;border-bottom:1px solid #2a323c;
       padding:14px 20px;z-index:10}
h1{margin:0 0 6px;font-size:19px}
.sub{color:#9aa7b5;font-size:13px}
.nav{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
.nav a{background:#232b35;border:1px solid #38434f;color:#e8ecf1;padding:6px 12px;
       border-radius:5px;font-size:13px;text-decoration:none}
.nav a:hover{background:#2c3641}
.nav a.cur{background:#2d6ca8;border-color:#3d84c6}
.grid{padding:16px 20px;display:flex;flex-direction:column;gap:16px}
.card{background:#171c23;border:1px solid #2a323c;border-radius:7px;padding:10px 12px}
.card.err{border-left:4px solid #d64545}
.card.res{border-left:4px solid #d69e45}
.card.ok{border-left:4px solid #3f9b57}
.hdr{display:flex;justify-content:space-between;align-items:baseline;
     gap:12px;flex-wrap:wrap;margin-bottom:7px}
.tag{font-weight:600;font-size:15px}
.metrics{color:#9aa7b5;font-size:12px;font-family:ui-monospace,monospace}
img{max-width:680px;width:100%;border-radius:4px;display:block;background:#000}
.note{color:#c9a45a;font-size:12px;margin:6px 0 0}
"""


def page(title, subtitle, cards, links, cur):
    nav = "".join(f'<a class="{"cur" if k == cur else ""}" href="{v}">{k}</a>'
                  for k, v in links)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{title}</title><style>{CSS}</style></head><body>
<header><h1>{title}</h1><div class="sub">{subtitle}</div>
<div class="nav">{nav}</div></header>
<div class="grid">
{''.join(cards)}
</div></body></html>"""


def card(rec):
    note = f'<p class="note">{rec["note"]}</p>' if rec.get("note") else ""
    return f"""<div class="card {rec['cls']}" id="{rec['tag']}">
  <div class="hdr"><span class="tag">{rec['tag']}</span>
    <span class="metrics">{rec['metrics']}</span></div>
  <img loading="lazy" src="data:image/{rec.get('fmt', 'jpeg')};base64,{rec['img']}"
       alt="{rec['tag']}">{note}
</div>"""


def main(a):
    here = Path(__file__).resolve().parent
    outdir = Path(a.output_dir) if a.output_dir else here / "QC"
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = a.csv
    if not csv_path:
        hits = sorted((here / "Tables").glob(
            f"electrode_positions_{a.network}_*.csv"))
        csv_path = str(hits[-1]) if hits else None
    valid = load_extraction(csv_path)

    log_path = a.log
    if not log_path:
        hits = sorted((here / "_tmp").glob(
            f"log_extract_coordinates_{a.network}_*.txt"))
        log_path = str(hits[-1]) if hits else None
    rescued = load_rescued(log_path)

    infs = find_inference(a.images, a.network)
    if not infs:
        raise SystemExit(f"No *_inference_{a.network}.nii.gz under {a.images}")

    print(f"\n{'=' * 62}")
    print("  Visual QC report")
    print(f"{'=' * 62}")
    print(f"  Network        : {a.network}")
    print(f"  Inference files: {len(infs):,}")
    print(f"  Extraction CSV : {csv_path or '(none)'}")
    print(f"  Extraction log : {log_path or '(none -- Rescued page empty)'}")
    print(f"  Accepted       : {len(valid):,}")
    print(f"  Rescued        : {len(rescued):,}")
    print(f"  Output         : {outdir}")
    print(f"{'=' * 62}\n")

    items = []
    for p in infs:
        tag = tag_from_inference(p, a.network)
        anat = p.replace(f"_PDw_inference_{a.network}.nii.gz", "_PDw.nii")
        if not os.path.isfile(anat):
            continue
        row = valid.get(tag)
        bits = []
        if row:
            for k, f in (("star_min_angle", "min angle {:.1f}°"),
                         ("star_radius_cv", "spoke CV {:.3f}"),
                         ("spoke_max_mm", "longest spoke {:.1f} mm")):
                v = fnum(row, k)
                if v is not None:
                    bits.append(f.format(v))
        if tag in rescued:
            grp, cls = "3_rescued", "res"
            note = f"recovered by post-processing — {rescued[tag]}"
        elif row is not None:
            grp, cls, note = "1_accepted", "ok", ""
        else:
            grp, cls, note = "2_errors", "err", ""
        items.append(dict(tag=tag, anat=anat, mask=p, cls=cls, group=grp,
                          project=project_of(tag), note=note,
                          metrics="  ·  ".join(bits) if bits
                          else "no coordinates extracted"))

    groups = {}
    for it in items:
        groups.setdefault(("all" if a.single_page else it["project"],
                           it["group"]), []).append(it)

    stamp = datetime.now().strftime("%Y%m%d")
    order = ["1_accepted", "2_errors", "3_rescued"]
    labels = {"1_accepted": "Accepted", "2_errors": "Errors",
              "3_rescued": "Rescued"}

    for scope in sorted({k[0] for k in groups}):
        links = [(f"{labels[g]} ({len(groups.get((scope, g), []))})",
                  f"qc_{a.network}_{scope}_{g}_{stamp}.html") for g in order]
        for g in order:
            gitems = sorted(groups.get((scope, g), []), key=lambda d: d["tag"])
            if a.limit:
                gitems = gitems[:a.limit]
            out = outdir / f"qc_{a.network}_{scope}_{g}_{stamp}.html"
            cur = f"{labels[g]} ({len(groups.get((scope, g), []))})"
            if not gitems:
                out.write_text(page(f"{labels[g]} — {a.network} — {scope}",
                                    "no images in this category", [], links, cur),
                               encoding="utf-8")
                print(f"  {out.name}       0 images")
                continue
            cards, done = [], 0
            for it in gitems:
                try:
                    it["fmt"], it["img"] = head_montage(
                        it["anat"], it["mask"], a.zoom, a.width)
                except Exception as exc:                        # noqa: BLE001
                    it["fmt"], it["img"] = "png", ""
                    it["note"] = f"could not render: {type(exc).__name__}: {exc}"
                cards.append(card(it))
                done += 1
                if done % 20 == 0:
                    print(f"    {scope}/{g}: {done}/{len(gitems)}", flush=True)
            out.write_text(page(
                f"{labels[g]} — {a.network} — {scope}",
                f"{len(gitems)} images · four electrode-focused rows per head "
                f"(sagittal, coronal, axial) · {a.zoom:.0f} mm field of view · "
                f"generated {datetime.now():%Y-%m-%d %H:%M}",
                cards, links, cur), encoding="utf-8")
            print(f"  {out.name}   {len(gitems):>5} images   "
                  f"{out.stat().st_size / 1024 / 1024:6.1f} MB")

    print("\n  Pages are cross-linked. Start with Rescued: those images failed")
    print("  the plausibility checks and were re-admitted by a heuristic, so")
    print("  they carry the highest risk in both directions.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", required=True)
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--log", default=None,
                    help="extraction log; its recovery lists define the "
                         "Rescued page. Default: newest for this network.")
    ap.add_argument("--output-dir", dest="output_dir", default=None)
    ap.add_argument("--single-page", dest="single_page", action="store_true",
                    help="one set of three pages for everything, instead of "
                         "one set per project")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--width", type=int, default=560,
                    help="montage width in px; drives the file size")
    ap.add_argument("--zoom", type=float, default=60.0,
                    help="side length in mm of the box shown around each "
                         "electrode")
    main(ap.parse_args())
