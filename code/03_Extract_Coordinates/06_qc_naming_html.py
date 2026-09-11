#!/usr/bin/env python3
"""
==============================================================
  06_qc_naming_html.py
==============================================================
  Visual verification of the electrode LABEL assignment made by
  02_Merge_tables_and_Corret_Naming_ablation_final.py.

  That script decides which extracted cluster is the anode and
  which are cathodes 1-3, by Hungarian matching against the
  intended (baseline) montage. This report shows each labelled
  electrode on the image so the assignment can be checked.

  WHY THIS IS WORTH CHECKING
  --------------------------
  The Hungarian assignment always returns the optimal pairing;
  it cannot tell whether the pairing is meaningful. Tested on
  synthetic montages it handles every permutation correctly,
  including cyclic ones, and a rigid offset of the baseline
  produces no spurious swap. Two situations defeat it:

    * a baseline recorded for the CONTRALATERAL montage. The
      mirrored target yields a total distance of about 210 mm
      for a 70 mm montage, which passes the 300 mm rejection
      gate, and the matching then relabels two electrodes to fit
      a target on the wrong side of the head. Nothing in the
      numbers flags this;
    * two electrodes extracted at nearly the same position,
      where the assignment between them is arbitrary.

  Both are obvious on inspection and invisible in the tables,
  which is what this report is for.

  THREE PAGES
    1_swapped     labels were reassigned; check the anode is the
                  central electrode and the cathodes surround it
    2_unchanged   assignment accepted as extracted
    3_skipped     no correction applied: missing electrodes, or
                  the distance gate rejected the baseline

  Each head shows three views with every electrode marked by its
  ASSIGNED label, and the intended baseline positions as hollow
  markers, so a mismatch between the two is visible directly.

--------------------------------------------------------------
  HOW TO USE
      python 06_qc_naming_html.py --network b_no_attention
      python 06_qc_naming_html.py --network b_no_attention --limit 40
==============================================================
"""

import os
import re
import csv
import glob
import base64
import argparse
from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_IMAGES = '/media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction'
PLANES = [("Sagittal", 0), ("Coronal", 1), ("Axial", 2)]
ELECTRODES = ["anode", "cathode1", "cathode2", "cathode3"]
COLOURS = {"anode": "#ff3b30", "cathode1": "#34c759",
           "cathode2": "#0a84ff", "cathode3": "#ffd60a"}
SHORT = {"anode": "A", "cathode1": "C1", "cathode2": "C2", "cathode3": "C3"}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_long(path):
    """Corrected long table -> extracted montages, baselines, and conditions.

    The baseline is keyed by (subject, condition), not by subject alone. Every
    subject has TWO baselines -- a target montage and a control montage on the
    opposite side of the head -- and 02 records in each extracted row which of
    them was applied. Keying by subject alone lets one silently overwrite the
    other, so roughly half the subjects would be drawn against a montage that
    was never used: the hollow markers would sit on the wrong side of the head
    and the residual would be meaningless, while the figure still looked
    perfectly well-formed.
    """
    d = pd.read_csv(path, low_memory=False)
    d["coordinates"] = pd.to_numeric(d["coordinates"], errors="coerce")
    if "condition" not in d.columns:
        d["condition"] = ""
    d["condition"] = d["condition"].fillna("").astype(str)

    # `condition` MUST be in the pivot index. Every subject has two baselines,
    # a target montage and a control montage on the opposite side of the head,
    # and leaving condition out of the index both collapses them into one row
    # and makes the column unavailable afterwards -- so each image would be
    # drawn against whichever montage happened to be read last.
    wide = d.pivot_table(
        index=["subject", "session", "run", "electrode", "condition"],
        columns="dimension", values="coordinates").reset_index()

    auto, base, cond_of, stim_of = {}, {}, {}, {}

    # `stim` (sham / active) is a property of the baseline and is dropped by
    # the pivot, which keeps only index columns and coordinate values, so it is
    # read from the raw table and keyed the same way the baseline is.
    if "stim" in d.columns:
        bl = d[d.session.eq("ses-baseline")]
        for sub, cond, st in zip(bl["subject"], bl["condition"], bl["stim"]):
            st = "" if pd.isna(st) else str(st).strip()
            if st:
                stim_of[(sub, str(cond))] = st

    for r in wide.itertuples():
        xyz = (r.X, r.Y, r.Z)
        if any(pd.isna(v) for v in xyz):
            continue
        cond = str(r.condition or "")
        if r.session == "ses-baseline":
            base.setdefault((r.subject, cond), {})[r.electrode] = xyz
        else:
            tag = f"{r.subject}_{r.session}_{r.run}"
            auto.setdefault(tag, {})[r.electrode] = xyz
            cond_of[tag] = cond
    return auto, base, cond_of, stim_of


def load_swapped(log_path):
    """{tag: swap description} from the correction log's 'Swaps:' lines."""
    out = {}
    if not log_path or not os.path.isfile(log_path):
        return out
    cur = None
    for line in open(log_path, encoding="utf-8", errors="replace"):
        m = re.search(r"(sub-\S+)\s*\|\s*(ses-\S+)\s*\|\s*(run-\S+)", line)
        if m:
            cur = f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
        if "Swaps:" in line and cur:
            out[cur] = line.split("Swaps:")[1].strip()
    return out


def find_image(images_root, network, tag):
    sub = tag.split("_")[0]
    ses = tag.split("_")[1]
    run = tag.split("_")[2]
    pat = os.path.join(images_root, sub, "unzipped",
                       f"r{sub}_{ses}_*{run}_PDw.nii")
    hits = glob.glob(pat)
    if not hits:
        return None, None
    anat = hits[0]
    mask = anat.replace("_PDw.nii", f"_PDw_inference_{network}.nii.gz")
    return anat, (mask if os.path.isfile(mask) else None)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def montage(anat_path, mask_path, coords, base_coords, width_px=760, dpi=100):
    """Three views with every electrode marked by its assigned label.

    Coordinates are stored in world space (voxel_to_corr_space applies the
    affine), so the inverse affine puts them back on the voxel grid the image
    is displayed in.
    """
    img = nib.load(anat_path)
    anat = np.asanyarray(img.dataobj).astype(np.float32)
    inv = np.linalg.inv(img.affine)
    vox = {e: nib.affines.apply_affine(inv, np.array(c))
           for e, c in coords.items()}
    bvox = {e: nib.affines.apply_affine(inv, np.array(c))
            for e, c in (base_coords or {}).items()}

    mask = (np.asanyarray(nib.load(mask_path).dataobj) > 0
            if mask_path else None)
    centre = np.mean(list(vox.values()), axis=0)

    lo, hi = np.percentile(anat, [1, 99.5])
    anat = np.clip((anat - lo) / max(hi - lo, 1e-6), 0, 1)

    fig, axes = plt.subplots(1, 3, dpi=dpi,
                             figsize=(width_px / dpi, width_px / 3 / dpi))
    for ax, (pname, axis) in zip(axes, PLANES):
        idx = int(np.clip(round(centre[axis]), 0, anat.shape[axis] - 1))
        ax.imshow(np.take(anat, idx, axis=axis).T, cmap="gray",
                  origin="lower", interpolation="nearest")
        if mask is not None:
            m = np.take(mask, idx, axis=axis).T
            if m.any():
                ax.contour(m, levels=[0.5], colors="#ff2d2d", linewidths=0.7)
        oth = [i for i in range(3) if i != axis]
        for e in ELECTRODES:
            if e in bvox:                      # intended position, hollow
                b = bvox[e]
                ax.plot(b[oth[0]], b[oth[1]], "o", mfc="none",
                        mec=COLOURS[e], mew=1.6, ms=13, zorder=4)
            if e in vox:                       # assigned label, filled
                v = vox[e]
                ax.plot(v[oth[0]], v[oth[1]], "o", color=COLOURS[e],
                        ms=8, zorder=5)
                ax.text(v[oth[0]] + 4, v[oth[1]] + 4, SHORT[e],
                        color=COLOURS[e], fontsize=9, fontweight="bold",
                        zorder=6, bbox=dict(boxstyle="round,pad=0.15",
                                            facecolor="#000000cc",
                                            edgecolor="none"))
        ax.set_title(pname, fontsize=8, color="w", pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#333")
    fig.subplots_adjust(0, 0, 1, 0.94, 0.012, 0)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02,
                facecolor="black")
    plt.close(fig)
    raw = buf.getvalue()
    try:
        from PIL import Image
        im = Image.open(BytesIO(raw)).convert("RGB")
        out = BytesIO()
        im.save(out, format="JPEG", quality=72, optimize=True)
        return "jpeg", base64.b64encode(out.getvalue()).decode("ascii")
    except Exception:                                           # noqa: BLE001
        return "png", base64.b64encode(raw).decode("ascii")


def star_check(coords):
    """Post-correction star verification.

    02_Merge... calls verify_anode_configuration BEFORE the Hungarian
    matching, on the uncorrected labels, and never re-checks afterwards. A
    swap can therefore leave a labelling in which the electrode called 'anode'
    is not the central one, and nothing in the pipeline notices.

    Returns (ok, central_label, min_angle_deg, radius_cv).
    """
    if len(coords) != 4:
        return False, None, float("nan"), float("nan")
    names = list(coords)
    pts = {k: np.array(v, float) for k, v in coords.items()}
    best = None
    for h in names:
        sp = [pts[o] - pts[h] for o in names if o != h]
        radii = [float(np.linalg.norm(s)) for s in sp]
        if min(radii) < 1e-6:
            continue
        ang = []
        for i in range(3):
            for j in range(i + 1, 3):
                ca = np.dot(sp[i], sp[j]) / (np.linalg.norm(sp[i]) *
                                             np.linalg.norm(sp[j]) + 1e-9)
                ang.append(float(np.degrees(np.arccos(np.clip(ca, -1, 1)))))
        cand = (h, min(ang), float(np.std(radii) / np.mean(radii)))
        if best is None or cand[1] > best[1]:
            best = cand
    if best is None:
        return False, None, float("nan"), float("nan")
    return best[0] == "anode", best[0], best[1], best[2]


def mirror_check(coords, base, affine, shape):
    """Detect a baseline recorded for the CONTRALATERAL montage.

    The Hungarian matching returns the optimal assignment whatever the target;
    it cannot tell that the target is on the wrong side of the head. For a
    70 mm montage a mirrored baseline gives a total distance near 210 mm,
    which is better than identity and passes the 300 mm rejection gate, so the
    correction is applied and recorded as a success.

    Two independent signals are used, because neither alone is conclusive:

      residual : mean distance from each labelled electrode to its baseline
                 counterpart. A correct match is a few millimetres to a few
                 tens; a mirrored one is roughly half the montage width.
      side     : whether the anode and its baseline lie on opposite sides of
                 the mid-sagittal plane, estimated from the image centre.
    """
    if not base or len(coords) != 4 or len(base) < 4:
        return None
    common = [e for e in ELECTRODES if e in coords and e in base]
    if len(common) < 4:
        return None
    resid = float(np.mean([np.linalg.norm(np.array(coords[e]) -
                                          np.array(base[e])) for e in common]))
    centre = nib.affines.apply_affine(affine, np.array(shape) / 2.0)
    ax = coords["anode"][0] - centre[0]
    bx = base["anode"][0] - centre[0]
    opposite = (ax * bx < 0) and (abs(ax) > 12) and (abs(bx) > 12)
    return dict(residual_mm=resid, opposite_side=bool(opposite))


def geometry_note(coords):
    """Is the electrode labelled 'anode' actually the central one?

    Checked independently of the correction: the central electrode of a 3x1
    montage is the one whose three spokes subtend the largest minimum angle.
    Disagreement does not prove the labelling wrong, but it is the single most
    useful thing to look at on a card.
    """
    if len(coords) != 4:
        return "fewer than four electrodes", True
    names = list(coords)
    pts = {k: np.array(v, float) for k, v in coords.items()}
    best, bn = -1.0, None
    for h in names:
        sp = [pts[o] - pts[h] for o in names if o != h]
        ang = []
        for i in range(3):
            for j in range(i + 1, 3):
                ca = np.dot(sp[i], sp[j]) / (np.linalg.norm(sp[i]) *
                                             np.linalg.norm(sp[j]) + 1e-9)
                ang.append(np.degrees(np.arccos(np.clip(ca, -1, 1))))
        if min(ang) > best:
            best, bn = min(ang), h
    if bn == "anode":
        return f"geometry agrees: anode is central (min angle {best:.0f}°)", False
    return (f"geometry DISAGREES: '{bn}' looks central, not the anode "
            f"(min angle {best:.0f}°)"), True


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;
     background:#11151a;color:#e8ecf1}
header{position:sticky;top:0;background:#171c23;border-bottom:1px solid #2a323c;
       padding:14px 20px;z-index:10}
h1{margin:0 0 6px;font-size:19px}.sub{color:#9aa7b5;font-size:13px}
.nav{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
.nav a{background:#232b35;border:1px solid #38434f;color:#e8ecf1;padding:6px 12px;
       border-radius:5px;font-size:13px;text-decoration:none}
.nav a.cur{background:#2d6ca8;border-color:#3d84c6}
.legend{margin-top:9px;font-size:12px;color:#9aa7b5}
.legend b{color:#e8ecf1}
.grid{padding:16px 20px;display:flex;flex-direction:column;gap:15px}
.card{background:#171c23;border:1px solid #2a323c;border-radius:7px;padding:10px 12px}
.card.warn{border-left:4px solid #d64545}
.card.ok{border-left:4px solid #3f9b57}
.hdr{display:flex;justify-content:space-between;align-items:baseline;gap:12px;
     flex-wrap:wrap;margin-bottom:7px}
.tag{font-weight:600;font-size:15px}
.meta{color:#9aa7b5;font-size:12px;font-family:ui-monospace,monospace}
img{max-width:820px;width:100%;border-radius:4px;display:block;background:#000}
.note{font-size:12px;margin:6px 0 0}.note.bad{color:#ff8b8b}.note.good{color:#8bdba0}
"""


def page(title, sub, cards, links, cur):
    nav = "".join(f'<a class="{"cur" if k == cur else ""}" href="{v}">{k}</a>'
                  for k, v in links)
    legend = ('<div class="legend">Filled marker = <b>assigned label</b>; '
              'hollow marker = <b>intended (baseline) position</b>. '
              '<span style="color:#ff3b30">A</span> anode, '
              '<span style="color:#34c759">C1</span> '
              '<span style="color:#0a84ff">C2</span> '
              '<span style="color:#ffd60a">C3</span> cathodes.</div>')
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<style>{CSS}</style></head><body>
<header><h1>{title}</h1><div class="sub">{sub}</div>
<div class="nav">{nav}</div>{legend}</header>
<div class="grid">{''.join(cards)}</div></body></html>"""


def card(r):
    cls = "warn" if r["warn"] else "ok"
    nc = "bad" if r["warn"] else "good"
    swap = f'<p class="note">swaps applied: <code>{r["swap"]}</code></p>' if r.get("swap") else ""
    return f"""<div class="card {cls}" id="{r['tag']}">
  <div class="hdr"><span class="tag">{r['tag']}</span>
    <span class="meta">{r['meta']}</span></div>
  <img loading="lazy" src="data:image/{r.get('fmt','jpeg')};base64,{r['img']}" alt="{r['tag']}">
  <p class="note {nc}">{r['note']}</p>{swap}
</div>"""


def main(a):
    here = Path(__file__).resolve().parent
    outdir = Path(a.output_dir) if a.output_dir else here / "QC_naming"
    outdir.mkdir(parents=True, exist_ok=True)

    src = a.input or (here / "Tables" /
                      f"corrected_electrode_positions_with_baseline_long_{a.network}.csv")
    if not os.path.isfile(src):
        raise SystemExit(f"Corrected long table not found: {src}\n"
                         f"Run 02_Merge_tables_and_Corret_Naming... --network {a.network}")
    log = a.log or (here / f"coordinate_correction_{a.network}.log")
    auto, base, cond_of, stim_of = load_long(src)
    n_multi = len({s for s, _ in base}) and sum(
        1 for s in {s for s, _ in base}
        if len([c for ss, c in base if ss == s]) > 1)
    if n_multi:
        print(f"  {n_multi} subject(s) have more than one baseline condition; "
              f"each image is drawn against the one recorded on its own rows.")
    swapped = load_swapped(log if os.path.isfile(str(log)) else None)

    print(f"\n{'=' * 62}\n  Naming-correction QC\n{'=' * 62}")
    print(f"  Network   : {a.network}")
    print(f"  Table     : {src}")
    print(f"  Log       : {log if os.path.isfile(str(log)) else '(none — page 1 will be empty)'}")
    print(f"  Images    : {len(auto):,}   with a matching baseline: "
          f"{sum(1 for t in auto if (t.split('_')[0], cond_of.get(t, '')) in base):,}")
    print(f"  Swapped   : {len(swapped):,}\n{'=' * 62}\n")

    items, rows = [], []
    missing_cond = 0
    for tag, coords in sorted(auto.items()):
        sub, ses, run = tag.split("_")[0], tag.split("_")[1], tag.split("_")[2]
        cond = cond_of.get(tag, "")
        stim = stim_of.get((sub, cond), "")
        if a.stim and stim not in a.stim:
            # Images whose baseline came from another stimulation folder, or
            # which have no baseline and therefore no folder, are skipped
            # rather than shown unlabelled.
            continue
        # The baseline for THIS image's condition. Falling back to any baseline
        # for the subject would reintroduce the bug this keying prevents, so a
        # missing one is reported instead.
        subject_base = base.get((sub, cond))
        if subject_base is None:
            avail = [c for s, c in base if s == sub]
            if avail:
                missing_cond += 1
                if missing_cond <= 5:
                    print(f"  {tag}: no '{cond}' baseline, though "
                          f"{'/'.join(sorted(avail))} exist(s). Drawn without "
                          f"an intended position rather than against the wrong "
                          f"montage.")
        anat, mask = find_image(a.images, a.network, tag)
        if not anat:
            continue

        ok_star, central, min_ang, rcv = star_check(coords)
        img = nib.load(anat)
        mir = mirror_check(coords, subject_base, img.affine,
                           img.header.get_data_shape()[:3])

        flags = []
        if len(coords) != 4:
            flags.append("fewer than four electrodes")
        elif not ok_star:
            flags.append(f"star check FAILED after correction: "
                         f"'{central}' is central, not the anode "
                         f"(min angle {min_ang:.0f}°)")
        if mir:
            if mir["opposite_side"]:
                flags.append(f"anode is on the OPPOSITE side of the midline "
                             f"from its baseline — contralateral baseline?")
            if mir["residual_mm"] > a.mirror_mm:
                flags.append(f"mean residual to baseline {mir['residual_mm']:.0f} mm "
                             f"> {a.mirror_mm:.0f} mm — the baseline may belong "
                             f"to a different montage")
        warn = bool(flags)
        note = ("; ".join(flags) if flags else
                f"star check passed: anode is central "
                f"(min angle {min_ang:.0f}°, spoke CV {rcv:.3f})")

        grp = ("1_swapped" if tag in swapped
               else "2_unchanged" if len(coords) == 4 else "3_skipped")
        meta = f"{len(coords)} electrode(s)"
        meta += (f"  ·  {cond} baseline" if subject_base
                 else f"  ·  no '{cond}' baseline")
        if stim:
            meta += f"  ·  {stim}"
        if mir:
            meta += f"  ·  residual {mir['residual_mm']:.1f} mm"
        items.append(dict(tag=tag, anat=anat, mask=mask, coords=coords,
                          base=subject_base, group=grp, note=note, warn=warn,
                          swap=swapped.get(tag, ""), meta=meta))

        rows.append(dict(
            network=a.network, subject=sub, session=ses, run=run,
            n_electrodes=len(coords), group=grp,
            swapped="yes" if tag in swapped else "no",
            swaps=swapped.get(tag, ""),
            star_ok="yes" if ok_star else "no",
            central_electrode=central or "",
            min_angle_deg=round(min_ang, 2) if min_ang == min_ang else "",
            spoke_cv=round(rcv, 4) if rcv == rcv else "",
            baseline_residual_mm=(round(mir["residual_mm"], 2) if mir else ""),
            opposite_side=("yes" if mir and mir["opposite_side"] else
                           ("no" if mir else "")),
            flagged="yes" if warn else "no",
            flags="; ".join(flags),
            coords=coords))

    # ── two coordinate tables ────────────────────────────────────────────
    # Separate columns per axis for analysis; a combined "(x, y, z)" string
    # per electrode for reading, because a person checking a cathode wants the
    # triple in one place rather than three columns to recombine mentally.
    stamp0 = datetime.now().strftime("%Y%m%d")
    base_cols = ["network", "subject", "session", "run", "n_electrodes",
                 "group", "swapped", "swaps", "star_ok", "central_electrode",
                 "min_angle_deg", "spoke_cv", "baseline_residual_mm",
                 "opposite_side", "flagged", "flags"]

    sep = outdir / f"electrode_labels_xyz_{a.network}_{stamp0}.csv"
    with open(sep, "w", newline="", encoding="utf-8") as fh:
        cols = base_cols + [f"{e}_{d}" for e in ELECTRODES for d in "XYZ"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = {k: r[k] for k in base_cols}
            for e in ELECTRODES:
                c = r["coords"].get(e)
                for i, d in enumerate("XYZ"):
                    out[f"{e}_{d}"] = round(c[i], 3) if c else ""
            w.writerow(out)

    comb = outdir / f"electrode_labels_combined_{a.network}_{stamp0}.csv"
    with open(comb, "w", newline="", encoding="utf-8") as fh:
        cols = base_cols + list(ELECTRODES)
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = {k: r[k] for k in base_cols}
            for e in ELECTRODES:
                c = r["coords"].get(e)
                out[e] = (f"({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})" if c else "")
            w.writerow(out)

    n_flag = sum(1 for r in rows if r["flagged"] == "yes")
    print(f"  {sep.name}   {len(rows)} rows")
    print(f"  {comb.name}   {len(rows)} rows")
    print(f"  flagged: {n_flag} of {len(rows)}\n")

    # A dedicated page for every flagged row, drawn from all three groups.
    # Flagged rows are scattered across Swapped, Unchanged and Skipped, and a
    # reviewer wants them in one place: these are the rows whose BASELINE is
    # doubtful, and therefore the ones to exclude from any deviation analysis.
    flagged = [dict(i, group="0_flagged") for i in items if i["warn"]]
    items = items + flagged

    order = ["0_flagged", "1_swapped", "2_unchanged", "3_skipped"]
    labels = {"0_flagged": "Flagged", "1_swapped": "Swapped",
              "2_unchanged": "Unchanged", "3_skipped": "Skipped"}
    # The filter goes into the filenames: a sham-only page must not be
    # mistakable for the whole set once it is sitting in a folder.
    stim_tag = ("_" + "-".join(sorted(a.stim))) if a.stim else ""
    stamp = datetime.now().strftime("%Y%m%d") + stim_tag
    groups = {g: [i for i in items if i["group"] == g] for g in order}
    links = [(f"{labels[g]} ({len(groups[g])})",
              f"qcname_{a.network}_{g}_{stamp}.html") for g in order]

    for g in order:
        gi = groups[g]
        # warnings first: a card flagged as geometrically inconsistent is the
        # one worth opening, and burying it behind hundreds of correct ones
        # defeats the purpose of the report
        gi.sort(key=lambda d: (not d["warn"], d["tag"]))
        if a.limit:
            gi = gi[:a.limit]
        out = outdir / f"qcname_{a.network}_{g}_{stamp}.html"
        cur = f"{labels[g]} ({len(groups[g])})"
        if not gi:
            out.write_text(page(f"{labels[g]} — {a.network}",
                                "no images in this category", [], links, cur),
                           encoding="utf-8")
            print(f"  {out.name}       0 images")
            continue
        cards = []
        for n, it in enumerate(gi, 1):
            try:
                it["fmt"], it["img"] = montage(it["anat"], it["mask"],
                                               it["coords"], it["base"],
                                               a.width)
            except Exception as exc:                            # noqa: BLE001
                it["fmt"], it["img"] = "png", ""
                it["note"] += f" — render failed: {type(exc).__name__}: {exc}"
            cards.append(card(it))
            if n % 20 == 0:
                print(f"    {g}: {n}/{len(gi)}", flush=True)
        n_warn = sum(1 for i in gi if i["warn"])
        sub = (f"{len(gi)} images whose baseline is doubtful — exclude these "
               f"from deviation analyses with 04_baseline_deviation.py "
               f"--exclude-note · generated {datetime.now():%Y-%m-%d %H:%M}"
               if g == "0_flagged" else
               f"{len(gi)} images · {n_warn} flagged where the geometry "
               f"disagrees with the assigned anode · generated "
               f"{datetime.now():%Y-%m-%d %H:%M}")
        out.write_text(page(f"{labels[g]} — {a.network}", sub, cards, links, cur),
                       encoding="utf-8")
        print(f"  {out.name}   {len(gi):>5} images   {n_warn:>4} flagged   "
              f"{out.stat().st_size / 1024 / 1024:6.1f} MB")

    if missing_cond:
        print(f"\n  {missing_cond} image(s) had no baseline for the condition "
              f"recorded on them. That points back at 02: it stamped a "
              f"condition\n  for which no montage was loaded.")

    flag_rows = [i for i in items if i["group"] == "0_flagged"]
    if flag_rows:
        fp = outdir / f"flagged_baselines_{a.network}_{stamp}.csv"
        with open(fp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["subject", "session", "run", "tag", "reason"])
            for i in sorted(flag_rows, key=lambda d: d["tag"]):
                parts = i["tag"].split("_")
                w.writerow(parts[:3] + [i["tag"], i["note"]])
        print(f"  {fp.name}   {len(flag_rows)} flagged image(s)")

    print("\n  Open the Flagged page first, and look for a filled marker that")
    print("  sits on the wrong side of the head from its hollow counterpart:")
    print("  that is the signature of a baseline recorded for the")
    print("  contralateral montage, which the distance gate does not catch.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", required=True)
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--input", default=None,
                    help="corrected long CSV; default: Tables/…_long_<network>.csv")
    ap.add_argument("--log", default=None,
                    help="coordinate_correction_<network>.log, whose 'Swaps:' "
                         "lines define the Swapped page")
    ap.add_argument("--output-dir", dest="output_dir", default=None)
    ap.add_argument("--stim", nargs="+", default=None,
                    choices=["sham", "active"], metavar="FOLDER",
                    help="show only images whose baseline came from these "
                         "stimulation folders, e.g. --stim active")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--mirror-mm", dest="mirror_mm", type=float, default=40.0,
                    help="mean distance to the baseline above which the "
                         "baseline is flagged as possibly belonging to a "
                         "different or contralateral montage")
    main(ap.parse_args())
