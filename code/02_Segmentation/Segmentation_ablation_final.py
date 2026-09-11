"""
==============================================================
  Segmentation_paper_models.py
==============================================================
  Author  : Filip Niemann
  Contact : filip.niemann@med.uni-greifswald.de

  Questions, bug reports, and feature requests are welcome --
  please reach out by e-mail.
--------------------------------------------------------------
  DESCRIPTION
  -----------
  Runs automated electrode segmentation inference on all
  subjects found under the images root directory, using one of
  the trained networks in code/Network/Paper_models/.

  For every coregistered Petra NIfTI file (rsub-*_PDw.nii)
  found inside a subject's unzipped/ folder, the script loads
  the selected model, performs segmentation, and saves the
  result next to the input as

      <name>_PDw_inference_<suffix>.nii.gz

  where <suffix> defaults to the network key of the chosen
  model (proposed, b_no_attention, a_baseline, c_reduced,
  d_increased) so that segmentations from different networks
  never overwrite each other and stay traceable.

  ARCHITECTURE IS NOT GUESSED. Each network key carries its
  exact constructor arguments, and the parameter count is
  checked against the value printed by the training run before
  the weights are loaded. A mismatch aborts rather than
  silently producing nonsense.

--------------------------------------------------------------
  HOW TO USE
  ----------
  1. List the models available in Paper_models/:

       python Segmentation_paper_models.py --list

  2. Run the proposed Attention U-Net:

       python Segmentation_paper_models.py \\
           --model paper_attention_unet_RU_FINAL_seed1001_proposed_best_metric_model.pth

     -> writes  *_PDw_inference_proposed.nii.gz

  3. Run the no-attention variant:

       python Segmentation_paper_models.py \\
           --model b_proposed_no_attention_RU_FINAL_seed1001_b_no_attention_best_metric_model.pth

     -> writes  *_PDw_inference_b_no_attention.nii.gz

  4. Override the output suffix:

       python Segmentation_paper_models.py --model <file> --suffix increased

     -> writes  *_PDw_inference_increased.nii.gz

  5. Use the GPU (much faster for large cohorts):

       python Segmentation_paper_models.py --model <file> --device cuda

  6. Other images root, overwrite existing outputs:

       python Segmentation_paper_models.py --model <file> \\
           --images /media/OtherDisk/derivatives/electrode_extraction \\
           --overwrite

  7. Show all options and current defaults:

       python Segmentation_paper_models.py --help

--------------------------------------------------------------
  NOTE ON INFERENCE MODE
  ----------------------
  Segmentation is done on the whole volume in one forward pass,
  exactly as in the original Segmentation.py. This is kept
  deliberately: it is what the published pipeline did, so
  results stay comparable across networks and across the old
  and new models. If you ever switch to sliding-window
  inference, switch it for ALL models at once or the comparison
  is void.
==============================================================
"""

import os
import sys
import glob
import time
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np
import nibabel as nib
import monai.transforms as tfms
from monai.networks.nets import AttentionUnet, UNet

# ============================================================
#  DEFAULT PATHS  --  change these if your layout moves,
#                     or override them at runtime via CLI args.
# ============================================================

DEFAULT_IMAGES_PATH = '/media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction'
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, '..', 'Network', 'Paper_models')
)
# ============================================================
#  NETWORK REGISTRY
#
#  One entry per manuscript Table 5 row. `params` is the count
#  printed by the corresponding training run and is verified at
#  load time. Constructor arguments are taken verbatim from the
#  training reports -- do not edit one without the other.
#
#  KEY ORDER MATTERS. Detection scans this list in order and
#  takes the first key found in the filename. 'b_no_attention'
#  must come before 'proposed', because the no-attention
#  checkpoint is named
#      b_proposed_no_attention_..._b_no_attention_...
#  which contains the substring 'proposed'.
# ============================================================

NETWORKS = [
    ("d_increased", dict(
        params=53_150_586,
        description="(d) Attention U-Net, increased widths 48-768",
        build=lambda: AttentionUnet(
            spatial_dims=3, in_channels=1, out_channels=2,
            channels=(48, 96, 192, 384, 768), strides=(2, 2, 2, 2),
            kernel_size=3, up_kernel_size=3, dropout=0.2),
    )),
    ("c_reduced", dict(
        params=5_909_130,
        description="(c) Attention U-Net, reduced widths 16-256",
        build=lambda: AttentionUnet(
            spatial_dims=3, in_channels=1, out_channels=2,
            channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
            kernel_size=3, up_kernel_size=3, dropout=0.2),
    )),
    ("b_no_attention", dict(
        params=7_912_874,
        description="(b) Proposed widths 32-512, NO attention gates",
        build=lambda: UNet(
            spatial_dims=3, in_channels=1, out_channels=2,
            channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
            kernel_size=3, up_kernel_size=3, num_res_units=0, dropout=0.2),
    )),
    ("a_baseline", dict(
        params=7_751_240,
        description="(a) Standard U-Net baseline, 64-512",
        build=lambda: UNet(
            spatial_dims=3, in_channels=1, out_channels=2,
            channels=(64, 128, 256, 512), strides=(2, 2, 2),
            kernel_size=3, up_kernel_size=3, num_res_units=0, dropout=0.0),
    )),
    ("proposed", dict(
        params=23_625_986,
        description="Proposed Attention U-Net, 32-512",
        build=lambda: AttentionUnet(
            spatial_dims=3, in_channels=1, out_channels=2,
            channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
            kernel_size=3, up_kernel_size=3, dropout=0.2),
    )),
]


# ============================================================
#  FUNCTIONS
# ============================================================

def detect_network_key(model_filename):
    """Return the registry key whose name appears in the filename.

    Scans NETWORKS in order, so more specific keys win. Raises if
    nothing matches -- better than defaulting to a wrong architecture.
    """
    name = os.path.basename(model_filename)
    for key, _spec in NETWORKS:
        if key in name:
            return key
    raise ValueError(
        f"Cannot infer the network architecture from '{name}'.\n"
        f"Expected the filename to contain one of: "
        f"{', '.join(k for k, _ in NETWORKS)}.\n"
        f"Pass --network explicitly to override."
    )


def list_models(model_dir):
    """Print the .pth files in model_dir with their detected architecture."""
    paths = sorted(glob.glob(os.path.join(model_dir, '*.pth')))
    if not paths:
        print(f"No .pth files found in {model_dir}")
        return
    print(f"\nModels in {model_dir}:\n")
    for p in paths:
        try:
            key = detect_network_key(p)
            spec = dict(NETWORKS)[key]
            info = f"{key}  --  {spec['description']}  ({spec['params']:,} params)"
        except ValueError:
            info = "UNRECOGNISED architecture (use --network)"
        print(f"  {os.path.basename(p)}\n      -> {info}\n")


def load_image(image_path):
    """Load and preprocess the input image, preserving the affine matrix."""
    nii_img = nib.load(image_path)
    affine  = nii_img.affine
    header  = nii_img.header

    transform = tfms.Compose([
        tfms.LoadImage(image_only=True),
        tfms.ScaleIntensity(),
        tfms.EnsureChannelFirst(),
    ])
    image = transform(image_path)

    return image, affine, header


def get_model(model_path, network_key, device, quiet=False):
    """Build the architecture for network_key and load the weights into it.

    Verifies the parameter count against the training report before loading,
    and loads with strict=True so any layer-name or shape mismatch raises
    instead of silently leaving parts of the network at random init.
    """
    spec  = dict(NETWORKS)[network_key]
    model = spec['build']().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if n_params != spec['params']:
        raise RuntimeError(
            f"Architecture check FAILED for '{network_key}':\n"
            f"  built     {n_params:,} parameters\n"
            f"  expected  {spec['params']:,} (from the training report)\n"
            f"The registry entry and the trained model have diverged. "
            f"Do not use these segmentations."
        )
    if not quiet:
        print(f"  Architecture : {spec['description']}")
        print(f"  Parameters   : {n_params:,}  (matches training report)")

    state = torch.load(model_path, map_location=device, weights_only=False)
    if hasattr(state, 'state_dict'):          # a pickled nn.Module, not a dict
        state = state.state_dict()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def segment_image(model, image, verbose=False):
    """Perform segmentation on the input image and return a 3D volume."""
    device = next(model.parameters()).device
    with torch.no_grad():
        input_tensor = torch.unsqueeze(torch.as_tensor(image).to(device), 0)
        output = model(input_tensor)
        if verbose:
            print(f"  Model output shape        : {tuple(output.shape)}")

    post_pred    = tfms.Compose([tfms.AsDiscrete(argmax=True)])
    segmentation = post_pred(output[0])
    segmentation = segmentation.squeeze(0)
    if verbose:
        print(f"  3D segmentation shape     : {tuple(segmentation.shape)}")

    return segmentation.cpu().numpy()


def save_segmentation(segmentation, output_path, affine):
    """Save the segmentation as a NIfTI file, preserving the affine matrix."""
    if segmentation.ndim != 3:
        raise ValueError(f"Segmentation must be 3D, but got shape: {segmentation.shape}")
    nib.save(nib.Nifti1Image(segmentation.astype(np.uint8), affine), output_path)


# ============================================================
#  MAIN
# ============================================================


# ============================================================
#  PARALLEL WORKERS  (CPU only)
#
#  Two independent knobs, and their PRODUCT is what consumes cores:
#      --jobs    N   separate processes, each segmenting a whole volume
#      --threads T   PyTorch intra-op threads inside each process
#  Total cores used ~= N * T.
#
#  Which split is best is an empirical question and depends on RAM, not
#  just core count: whole-volume 3D inference holds several GB of
#  activations per process, so N processes need N times that. Start with
#  --jobs 1 --threads 12, watch RSS in `top`, then raise --jobs only if
#  there is headroom. See --benchmark.
# ============================================================

_W = {}


def _init_worker(model_path, network_key, threads):
    """Runs once per worker process: pin thread count, load the model."""
    import torch as _t
    _t.set_num_threads(max(1, int(threads)))
    try:
        _t.set_flush_denormal(True)      # denormals are slow on CPU
    except Exception:
        pass
    _W['model'] = get_model(model_path, network_key, _t.device('cpu'), quiet=True)


def _segment_one(task):
    """Segment one volume. Returns (status, input_path, message)."""
    nifti_file, out_path = task
    try:
        image, affine, _ = load_image(nifti_file)
        seg = segment_image(_W['model'], image, verbose=False)
        save_segmentation(seg, out_path, affine)
        return ('ok', nifti_file, None)
    except Exception as exc:                       # noqa: BLE001
        return ('fail', nifti_file, f"{type(exc).__name__}: {exc}")


def _collect_tasks(root_images, out_tail, overwrite):
    """Walk the tree once and split into work / already-done."""
    tasks, skipped = [], 0
    for subject_dir in sorted(glob.glob(os.path.join(root_images, 'sub-*'))):
        unzipped = os.path.join(subject_dir, 'unzipped')
        for f in sorted(glob.glob(os.path.join(
                unzipped, 'rsub-*_ses-*_acq-petra_run-*_PDw.nii'))):
            out = f.replace('_PDw.nii', out_tail)
            if os.path.exists(out) and not overwrite:
                skipped += 1
            else:
                tasks.append((f, out))
    return tasks, skipped


def main(args):

    if args.list:
        list_models(args.model_dir)
        return

    if not args.model_name:
        raise SystemExit(
            "No model selected. Use --model <filename>, or --list to see "
            "what is available in the model directory."
        )

    root_images = args.images
    model_path  = os.path.join(args.model_dir, args.model_name)

    network_key = args.network or detect_network_key(args.model_name)
    suffix      = args.suffix or network_key
    out_tail    = f'_PDw_inference_{suffix}.nii.gz'

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    n_cpu   = os.cpu_count() or 1
    jobs    = max(1, int(args.jobs))
    threads = int(args.threads) if args.threads else max(1, args.cpu_budget // jobs)
    used    = jobs * threads

    print("\n" + "=" * 62)
    print("  Inference -- configuration")
    print("=" * 62)
    print(f"  Images path  : {root_images}")
    print(f"  Model dir    : {args.model_dir}")
    print(f"  Model name   : {args.model_name}")
    print(f"  Network key  : {network_key}"
          f"{'  (from --network)' if args.network else '  (detected from filename)'}")
    print(f"  Output suffix: {out_tail}")
    print(f"  Device       : {device}")
    print(f"  Overwrite    : {args.overwrite}")
    if device.type == 'cpu':
        print(f"  CPU parallel : {jobs} process(es) x {threads} thread(s) "
              f"= {used} of {n_cpu} cores")
        if used > n_cpu:
            print(f"  WARNING      : {used} > {n_cpu} available cores -- "
                  f"oversubscription will SLOW the run down.")
    print("=" * 62 + "\n")

    if not os.path.isdir(root_images):
        raise FileNotFoundError(f"Images path not found: {root_images}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Thread limits must be in the environment BEFORE a worker imports torch,
    # otherwise the OpenMP pool is already sized and set_num_threads() only
    # partly takes effect. Children inherit this env at spawn time.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)

    tasks, n_skipped = _collect_tasks(root_images, out_tail, args.overwrite)
    n_total = len(tasks) + n_skipped
    if n_total == 0:
        print(f"WARNING: no rsub-*_PDw.nii files found under {root_images}")
        return
    print(f"  {n_total} image(s) found: {len(tasks)} to segment, "
          f"{n_skipped} already done.\n")

    if args.benchmark:
        tasks = tasks[:args.benchmark]
        print(f"  BENCHMARK MODE: processing only the first {len(tasks)} "
              f"image(s) to time this configuration.\n")

    n_done = n_failed = 0
    t0 = time.time()

    def _progress(i, total, path, status, msg=None):
        el = time.time() - t0
        rate = i / el if el > 0 else 0
        eta = (total - i) / rate if rate > 0 else 0
        tag = "OK  " if status == 'ok' else "FAIL"
        line = (f"  [{i}/{total}] {tag} {os.path.basename(path)[:52]:<52} "
                f"{rate*60:5.1f} img/min  ETA {int(eta//3600)}h{int(eta%3600//60):02d}m")
        print(line if status == 'ok' else line + f"\n       {msg}", flush=True)

    if jobs == 1 or device.type != 'cpu':
        # Sequential path. Behaviour identical to the single-process version.
        print(f"Loading model: {model_path}")
        torch.set_num_threads(threads)
        model = get_model(model_path, network_key, device)
        print("Model loaded.\n")
        for i, (nifti_file, out_path) in enumerate(tasks, 1):
            try:
                image, affine, _ = load_image(nifti_file)
                seg = segment_image(model, image, verbose=args.verbose)
                save_segmentation(seg, out_path, affine)
                n_done += 1
                _progress(i, len(tasks), nifti_file, 'ok')
            except Exception as exc:                    # noqa: BLE001
                n_failed += 1
                _progress(i, len(tasks), nifti_file, 'fail',
                          f"{type(exc).__name__}: {exc}")
    else:
        print(f"Starting {jobs} worker process(es), {threads} torch thread(s) each.")
        print("Each worker loads its own copy of the model; watch RSS in `top` "
              "and lower --jobs if memory is tight.\n")
        ctx = mp.get_context("spawn")   # safer than fork once torch is imported
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx,
                                 initializer=_init_worker,
                                 initargs=(model_path, network_key, threads)) as ex:
            futures = {ex.submit(_segment_one, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                status, path, msg = fut.result()
                if status == 'ok':
                    n_done += 1
                else:
                    n_failed += 1
                _progress(i, len(tasks), path, status, msg)

    el = time.time() - t0
    print(f"\n{'=' * 62}")
    print(f"  Done.  segmented: {n_done}   skipped: {n_skipped}   failed: {n_failed}")
    print(f"  Suffix written: {out_tail}")
    print(f"  Wall clock: {int(el//3600)}h {int(el%3600//60)}m {int(el%60)}s"
          f"   ({n_done/el*60:.1f} img/min)" if n_done else "")
    if args.benchmark and n_done:
        full = (len(_collect_tasks(root_images, out_tail, args.overwrite)[0]))
        print(f"  Extrapolated for {full} remaining images: "
              f"{full*el/n_done/3600:.1f} h at this setting.")
    print(f"{'=' * 62}\n")


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run electrode segmentation inference using a Paper_models network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--images",
        default=DEFAULT_IMAGES_PATH,
        help="Root folder containing subject directories (sub-*).",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        dest="model_dir",
        help="Directory that contains the Paper_models .pth files.",
    )
    parser.add_argument(
        "--model", "--model-name",
        default=None,
        dest="model_name",
        help="Filename of the model weights (.pth) inside --model-dir.",
    )
    parser.add_argument(
        "--network",
        default=None,
        choices=[k for k, _ in NETWORKS],
        help="Force the architecture instead of detecting it from the filename.",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Output suffix; written as *_PDw_inference_<suffix>.nii.gz. "
             "Defaults to the network key.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | cpu | cuda | cuda:0 ...",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing inference files if they already exist.",
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=1,
        help="Number of parallel worker PROCESSES (CPU only). Each holds its "
             "own copy of the model and its own activations, so RAM scales "
             "with this. Start at 1, check RSS, then raise.",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="PyTorch intra-op threads per worker. Default: cpu-budget // jobs.",
    )
    parser.add_argument(
        "--cpu-budget", dest="cpu_budget", type=int, default=12,
        help="Total cores this job may use. Only used to derive --threads "
             "when --threads is not given explicitly.",
    )
    parser.add_argument(
        "--benchmark", type=int, default=0, metavar="N",
        help="Process only the first N images, then report throughput and "
             "extrapolate the full run. Use this to tune --jobs/--threads.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print tensor shapes for every image.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the models in --model-dir with their detected architecture, then exit.",
    )

    main(parser.parse_args())
