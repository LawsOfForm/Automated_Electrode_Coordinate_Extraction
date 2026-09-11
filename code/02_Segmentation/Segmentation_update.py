"""
==============================================================
  Segmentation.py
==============================================================
  Author  : Filip Niemann
  Contact : filip.niemann@med.uni-greifswald.de
  
  Questions, bug reports, and feature requests are welcome —
  please reach out by e-mail.
--------------------------------------------------------------
  DESCRIPTION
  -----------
  Runs automated electrode segmentation inference on all
  subjects found under the images root directory.
  For every coregistered Petra NIfTI file (rsub-*_PDw.nii)
  found inside a subject's unzipped/ folder, the script loads
  the AttentionUnet model, performs segmentation, and saves
  the result as *_PDw_inference.nii.gz in the same folder.

  Trained networks can be downloaded from here (adapt the path to the network):
  https://nextcloud.uni-greifswald.de/index.php/s/HmtA4wtqkbkaqy7?dir=/Networks

--------------------------------------------------------------
  HOW TO USE
  ----------
  1. Basic run (uses all default paths):

       python Segmentation.py

  2. Overwrite existing inference files:

       python Segmentation.py --overwrite

  3. Use a different model file:

       python Segmentation.py \
           --model-name best_metric_model_newrun.pth

  4. Use a different model directory:

       python Segmentation.py \
           --model-dir /path/to/my/models \
           --model-name my_model.pth

  5. Use a different images root folder:

       python Segmentation.py \
           --images /media/OtherDisk/derivatives/electrode_extraction

  6. Combine any of the above — all flags are optional and
     fall back to the defaults defined at the top of this file
     if not provided:

       python Segmentation.py \
           --images /media/OtherDisk/subjects \
           --model-dir /home/user/models \
           --model-name my_model.pth \
           --overwrite

  7. Show all options and current defaults:

       python Segmentation.py --help

--------------------------------------------------------------
  DEFAULT PATHS (edit at top of file if your layout changes)
  -----------------------------------------------------------
  Images root : /media/MeMoSLAP_Subjects/derivatives/
                    automated_electrode_extraction
  Model dir   : /media/Data03/Projects/
                    Automated_Electrode_Extraction/code/Network/models/
  Model name  : best_metric_model_0716_1903_5level_Adamax_72tr.pth
==============================================================
"""

import os
import glob
import argparse
import torch
import monai.transforms as tfms
from monai.networks.nets import AttentionUnet
import nibabel as nib
import numpy as np

# ============================================================
#  DEFAULT PATHS  —  change these if your layout moves,
#                    or override them at runtime via CLI args.
# ============================================================

DEFAULT_IMAGES_PATH = '/media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction'
DEFAULT_MODEL_DIR   = '/media/Data03/Projects/Automated_Electrode_Extraction/code/Network/models/'
DEFAULT_MODEL_NAME  = 'best_metric_model_0716_1903_5level_Adamax_72tr.pth'


# ============================================================
#  FUNCTIONS
# ============================================================

def load_image(image_path):
    """Load and preprocess the input image, preserving the affine matrix.

    Normalisation matches the training pipeline exactly:
      NormalizeIntensity(nonzero=True, channel_wise=True)
    i.e. zero-mean / unit-variance computed only over non-zero voxels,
    applied per channel.  The old ScaleIntensity() (min-max to [0,1])
    produced a different input distribution and degraded inference quality.
    """
    nii_img = nib.load(image_path)
    affine  = nii_img.affine
    header  = nii_img.header

    transform = tfms.Compose([
        tfms.LoadImage(image_only=True),
        tfms.EnsureChannelFirst(),
        tfms.NormalizeIntensity(nonzero=True, channel_wise=True),  # matches training
    ])
    image = transform(image_path)

    return image, affine, header


def _remap_state_dict(state_dict):
    """Remap state dict keys saved by an older MONAI AttentionUnet to the
    key layout expected by current MONAI.

    Older MONAI stored the bottleneck block as a Sequential with two named
    children — index 0 (the conv block) and index 1 (the attention+upconv
    decoder block):
        ...1.submodule.0.conv.0.conv.weight          (bottleneck conv)
        ...1.submodule.1.attention.W_g.0.conv.weight (attention gate)
        ...1.submodule.1.upconv.up.conv.weight        (up-conv)
        ...1.submodule.1.submodule.conv.0.conv.weight (next decoder level)

    Current MONAI flattened the bottleneck conv under a bare .conv key:
        ...1.submodule.conv.0.conv.weight

    This function renames the conv-block keys (submodule.0.conv → submodule.conv)
    and strips the extra decoder-level wrapping (submodule.1.submodule →
    submodule) so the remaining keys match the current layout.
    All other keys are left unchanged.
    """
    import re
    new_sd = {}
    for k, v in state_dict.items():
        # Rule 1 — bottleneck conv block:
        #   ...submodule.0.conv...  →  ...submodule.conv...
        k2 = re.sub(r'(submodule)\.0\.(conv)', r'\1.\2', k)
        # Rule 2 — next decoder level wrapped inside submodule.1.submodule:
        #   ...submodule.1.submodule...  →  ...submodule...
        k2 = re.sub(r'(submodule)\.1\.submodule', r'\1', k2)
        # Keys that still contain ".1.attention", ".1.upconv", ".1.merge"
        # belong to the attention gate / up-conv of the *current* level —
        # those key names are unchanged in current MONAI, so leave them.
        new_sd[k2] = v
    return new_sd


def get_model(model_path):
    """Load the trained AttentionUnet model.

    Architecture:
      channels = (32, 64, 128, 256, 512, 1024)  — 6-level encoder
      strides  = (2, 2, 2, 2, 2)                — 5 downsampling steps

    The .pth file was saved with an older version of MONAI whose
    AttentionUnet used a slightly different internal key layout for the
    bottleneck block.  _remap_state_dict() translates the old keys to the
    names expected by the current MONAI before loading, so no MONAI
    downgrade is required.
    """
    device = torch.device("cpu")
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512, 1024),  # 6-level encoder — matches training
        strides=(2, 2, 2, 2, 2),                  # 5 strides       — matches training
        kernel_size=3,
        up_kernel_size=3,
        dropout=0.2,
    ).to(device)

    raw_sd     = torch.load(model_path, map_location=device, weights_only=False)
    remapped_sd = _remap_state_dict(raw_sd)

    missing, unexpected = model.load_state_dict(remapped_sd, strict=False)

    # Filter out harmless num_batches_tracked keys (scalar counters added by
    # older PyTorch BatchNorm; current PyTorch does not store them in the
    # state dict by default and they have no effect on inference).
    real_missing    = [k for k in missing    if 'num_batches_tracked' not in k]
    real_unexpected = [k for k in unexpected if 'num_batches_tracked' not in k]

    if real_missing or real_unexpected:
        raise RuntimeError(
            f"State dict mismatch after remapping.\n"
            f"  Missing keys    : {real_missing}\n"
            f"  Unexpected keys : {real_unexpected}\n"
            f"The .pth was likely saved with a MONAI version whose "
            f"AttentionUnet layout differs more than the remapper handles."
        )

    if missing or unexpected:
        print(f"  Note: ignored {len(missing)} 'num_batches_tracked' key(s) "
              f"— harmless, no effect on inference.")

    model.eval()
    return model


def segment_image(model, image):
    """Perform segmentation on the input image and return a 3D volume."""
    device = next(model.parameters()).device
    with torch.no_grad():
        input_tensor = torch.unsqueeze(torch.as_tensor(image).to(device), 0)
        output = model(input_tensor)
        print(f"  Model output shape       : {output.shape}")

    post_pred    = tfms.Compose([tfms.AsDiscrete(argmax=True)])
    segmentation = post_pred(output[0])
    print(f"  Segmentation after argmax : {segmentation.shape}")

    segmentation = segmentation.squeeze(0)
    print(f"  3D segmentation shape     : {segmentation.shape}")

    return segmentation.cpu().numpy()


def save_segmentation(segmentation, output_path, affine):
    """Save the segmentation as a NIfTI file, preserving the affine matrix."""
    if segmentation.ndim != 3:
        raise ValueError(f"Segmentation must be 3D, but got shape: {segmentation.shape}")
    nib.save(nib.Nifti1Image(segmentation.astype(np.uint8), affine), output_path)


# ============================================================
#  MAIN
# ============================================================

def main(args):

    # ── Resolve paths ─────────────────────────────────────────
    root_images = args.images
    model_path  = os.path.join(args.model_dir, args.model_name)

    print("\n" + "=" * 60)
    print("  Inference — configuration")
    print("=" * 60)
    print(f"  Images path : {root_images}")
    print(f"  Model dir   : {args.model_dir}")
    print(f"  Model name  : {args.model_name}")
    print(f"  Model path  : {model_path}")
    print(f"  Overwrite   : {args.overwrite}")
    print("=" * 60 + "\n")

    # ── Validate paths before doing any work ──────────────────
    if not os.path.isdir(root_images):
        raise FileNotFoundError(f"Images path not found: {root_images}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # ── Load model once ───────────────────────────────────────
    print(f"Loading model: {model_path} ...")
    model = get_model(model_path)
    print("Model loaded.\n")

    # ── Loop over subjects ────────────────────────────────────
    subject_dirs = sorted(glob.glob(os.path.join(root_images, 'sub-*')))
    if not subject_dirs:
        print(f"WARNING: No subject directories found in {root_images}")
        return

    for subject_dir in subject_dirs:
        subject      = os.path.basename(subject_dir)
        unzipped_dir = os.path.join(subject_dir, 'unzipped')

        nifti_files = sorted(glob.glob(
            os.path.join(unzipped_dir, 'rsub-*_ses-*_acq-petra_run-*_PDw.nii')
        ))

        if not nifti_files:
            print(f"  {subject}: no matching NIfTI files in unzipped/, skipping.")
            continue

        print(f"\n{'─' * 60}")
        print(f"  Subject: {subject}  ({len(nifti_files)} file(s))")
        print(f"{'─' * 60}")

        for nifti_file in nifti_files:
            output_segmentation_path = nifti_file.replace('_PDw.nii', '_PDw_inference.nii.gz')

            if os.path.exists(output_segmentation_path) and not args.overwrite:
                print(f"  Skipping (already exists): {os.path.basename(nifti_file)}")
                continue

            print(f"  Processing: {os.path.basename(nifti_file)}")
            image, affine, _ = load_image(nifti_file)

            segmentation = segment_image(model, image)
            print(f"  Segmentation shape: {segmentation.shape}")

            save_segmentation(segmentation, output_segmentation_path, affine)
            print(f"  Saved: {output_segmentation_path}")

    print("\nDone.\n")


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on NIfTI images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # shows defaults in --help
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
        help="Directory that contains the model .pth file.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        dest="model_name",
        help="Filename of the model weights (.pth).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing inference files if they already exist.",
    )

    args = parser.parse_args()
    main(args)