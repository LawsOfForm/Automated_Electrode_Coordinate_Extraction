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

  Trained networks can be downloaded form here (adapt the path to the network):
  https://nextcloud.uni-greifswald.de/index.php/s/HmtA4wtqkbkaqy7?dir=/Networks

--------------------------------------------------------------
  HOW TO USE
  ----------
  1. Basic run (uses all default paths):

       python Segmentation.py

  2. Overwrite existing inference files:

       python Segmentation.py --overwrite

  3. Use a different model file:

       python Segmentation.py \\
           --model-name best_metric_model_newrun.pth

  4. Use a different model directory:

       python Segmentation.py \\
           --model-dir /path/to/my/models \\
           --model-name my_model.pth

  5. Use a different images root folder:

       python Segmentation.py \\
           --images /media/OtherDisk/derivatives/electrode_extraction

  6. Combine any of the above — all flags are optional and
     fall back to the defaults defined at the top of this file
     if not provided:

       python Segmentation.py \\
           --images /media/OtherDisk/subjects \\
           --model-dir /home/user/models \\
           --model-name my_model.pth \\
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


def get_model(model_path):
    """Load the trained model."""
    device = torch.device("cpu")
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        kernel_size=3,
        up_kernel_size=3,
        dropout=0.2,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    return model


def segment_image(model, image):
    """Perform segmentation on the input image and return a 3D volume."""
    device = next(model.parameters()).device
    with torch.no_grad():
        input_tensor = torch.unsqueeze(torch.as_tensor(image).to(device), 0)
        output = model(input_tensor)
        print(f"  Model output shape       : {output.shape}")

    post_pred   = tfms.Compose([tfms.AsDiscrete(argmax=True)])
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