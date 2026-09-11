"""
Ablation variant (c) — Attention U-Net at REDUCED channel depth (16–256).

Purpose
-------
This is the proposed 3D Attention U-Net with every channel width halved:
(16, 32, 64, 128, 256) instead of (32, 64, 128, 256, 512). Everything else —
number of levels, strides, kernel sizes, dropout, attention gates — is
identical to the proposed model, so the ONLY thing this variant changes is
network width. Together with the proposed model (32–512) and the increased
variant (48–768) it gives a three-point channel-depth sweep, which is what
lets the manuscript claim the proposed width sits near a practical optimum
rather than merely asserting it from two points.

Corresponds to row (c) of Table 5 in the manuscript
("Att, reduced chan", 16–256).

Run
---
    python c_reduced_model_paper_bioarxiv.py
    python c_reduced_model_paper_bioarxiv.py --dataset HGW --folds 5 --fold 0
"""

from __future__ import annotations

from monai.networks.nets import AttentionUnet

from ablation_utils import (
    build_common_argparser,
    run_experiment_from_args,
)


# --------------------------------------------------------------------------- #
# Experiment identity
# --------------------------------------------------------------------------- #
EXPERIMENT_TAG = "attention_unet_reduced"
EXPERIMENT_DESCRIPTION = (
    "Ablation variant (c): 3D Attention U-Net with reduced channel depths "
    "(16, 32, 64, 128, 256). Identical to the proposed model in every other "
    "respect — five hierarchical levels, stride-2 down-sampling, 3x3x3 "
    "kernels, attention-gated skip connections, dropout 0.2, single-channel "
    "grayscale input -> 2-channel (background / electrode) output. The only "
    "difference from the proposed model is that every channel width is "
    "halved, isolating the effect of network width on segmentation "
    "performance."
)


# --------------------------------------------------------------------------- #
# Model factory — the ONLY architecture definition in this file.
# Widths are half of the proposed model at every level; levels, strides,
# kernels and dropout are copied from c_normal_model_paper_bioarxiv.py so the
# comparison isolates width alone.
# --------------------------------------------------------------------------- #
def build_model():
    """Return the reduced-width 3D Attention U-Net (16-256)."""
    return AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        kernel_size=3,
        up_kernel_size=3,
        dropout=0.2,
    )


# Recorded in the report's "Model configuration" table. Mirrors build_model().
MODEL_CONFIG_SUMMARY = {
    "architecture": "AttentionUnet (MONAI) — reduced channel depth",
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 2,
    "channels": (16, 32, 64, 128, 256),
    "strides": (2, 2, 2, 2),
    "kernel_size": 3,
    "up_kernel_size": 3,
    "dropout": 0.2,
}


def main():
    parser = build_common_argparser(description=EXPERIMENT_DESCRIPTION)
    # Same schedule as the other c_* variants so the comparison is fair.
    parser.set_defaults(max_iterations=50_000)
    args = parser.parse_args()

    run_experiment_from_args(
        args,
        experiment_tag=EXPERIMENT_TAG,
        experiment_description=EXPERIMENT_DESCRIPTION,
        build_model=build_model,
        model_config_summary=MODEL_CONFIG_SUMMARY,
    )


if __name__ == "__main__":
    main()
