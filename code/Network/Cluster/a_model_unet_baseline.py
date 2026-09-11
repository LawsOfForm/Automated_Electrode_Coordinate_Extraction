"""
Ablation (a) — Standard U-Net baseline.

Purpose
-------
Plain MONAI `UNet` (NO Attention Gates) with the *canonical* U-Net channel
configuration (32, 64, 128, 256, 512). This is the textbook reference model
the reviewer asked for as a baseline against which the proposed
Attention-U-Net variant is compared.

Run
---
    python a_model_unet_baseline.py
"""

from monai.networks.nets import UNet


def build_model():
    # Canonical 4-level U-Net following the original Ronneberger et al. (2015)
    # channel progression starting at 64, adapted to 3D. No attention gates,
    # no custom hyper-parameters — the textbook reference architecture.
    # We pick the 64-base configuration so this baseline is *meaningfully*
    # different from variant (b), which keeps the proposed model's
    # 32-base channel schedule.
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(64, 128, 256, 512),
        strides=(2, 2, 2),
        num_res_units=0,           # vanilla conv blocks, no residual units
        kernel_size=3,
        up_kernel_size=3,
        dropout=0.0,               # no dropout — pure baseline
    )


MODEL_CONFIG = {
    "architecture": "MONAI UNet — classic Ronneberger-style baseline (NO attention)",
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 2,
    "channels": "(64, 128, 256, 512)",
    "strides": "(2, 2, 2)",
    "num_res_units": 0,
    "kernel_size": 3,
    "up_kernel_size": 3,
    "dropout": 0.0,
}

DESCRIPTION = (
    "Standard U-Net baseline — a vanilla 4-level encoder/decoder with the "
    "classic Ronneberger-style (64, 128, 256, 512) channel progression, no "
    "attention gates, no residual units, no dropout. This corresponds to the "
    "first ablation requested by the reviewer ('a standard U-Net baseline') "
    "and is deliberately *not* aligned with the proposed model's "
    "channel-depth choices, so that variant (a) tests the entire proposed "
    "architecture against a textbook reference while variant (b) isolates "
    "the contribution of the Attention Gates alone. All other training "
    "settings (loss, optimiser, scheduler, augmentations, data split) are "
    "kept identical to the proposed model."
)


if __name__ == "__main__":
    from ablation_utils import build_common_argparser, run_experiment_from_args
    args = build_common_argparser(description=__doc__).parse_args()
    run_experiment_from_args(
        args,
        experiment_tag="a_unet_baseline",
        experiment_description=DESCRIPTION,
        build_model=build_model,
        model_config_summary=MODEL_CONFIG,
    )
