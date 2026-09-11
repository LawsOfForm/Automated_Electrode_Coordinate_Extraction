"""
Ablation (b) — Proposed model WITHOUT Attention Gates.

Purpose
-------
This is the proposed Attention-U-Net stripped of its Attention Gates: a plain
MONAI `UNet` that *otherwise* keeps the proposed configuration -- same
channel-depth schedule, same strides, same kernel sizes, same dropout. Used
together with the proposed model it isolates the contribution of the
Attention Gates themselves.

Why this script was rewritten
-----------------------------
The previous version of this ablation used `num_res_units=2`, which inserts
residual sums inside each conv block. MONAI's `AttentionUnet` uses two-conv
blocks WITHOUT residual sums, so the old (b) was actually testing "remove
attention AND add residuals" simultaneously -- not a clean attention-only
ablation.

It also diverged in training (50 000 iters at lr=0.005, final test Dice
~0.001 -- see `b_proposed_no_attention_report.md`). Two factors made
that likely:

  1. AttentionUnet's attention gates act as a mild regulariser. The plain
     UNet at the same lr=0.005 destabilises early; the loss collapses
     before iter 10 000 and never recovers.
  2. With `num_res_units=2` the effective parameter count and gradient
     flow differ from the proposed model anyway, so any conclusion drawn
     from the broken comparison is suspect on two grounds.

The fixes applied here:
  - `num_res_units=0` so the block topology matches AttentionUnet (the
    only architectural difference vs the proposed model is now genuinely
    the attention gates on the skip connections).
  - Default learning rate dropped to 0.001 (vs 0.005 for the proposed
    model). The plain UNet doesn't have attention's regularisation cushion,
    and 1e-3 is the standard MONAI UNet learning rate. We accept that this
    is a deliberate departure from the "everything identical to the
    proposed run" mantra: training divergence at lr=5e-3 makes the
    architectural comparison meaningless, so a stable training recipe
    matters more than literal LR equality. The report records the LR used.
  - The sanity check in `ablation_utils.py` (added when this rewrite
    happened) will catch any future divergence within ~10 000 iters
    instead of wasting the remaining 40 000.

Run
---
    python b_model_proposed_no_attention.py

Override LR back to the proposed model's value if you want to reproduce
the failed run for the paper supplementary:

    python b_model_proposed_no_attention.py --lr 0.005
"""

from monai.networks.nets import UNet


def build_model():
    # Channels, strides, kernel sizes, dropout match the proposed
    # AttentionUnet *exactly*. num_res_units=0 also matches AttentionUnet's
    # block topology. The ONLY architectural difference vs. the proposed
    # model is the absence of attention gates on the skip connections.
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=0,           # matches AttentionUnet block topology
        kernel_size=3,
        up_kernel_size=3,
        dropout=0.2,
    )


MODEL_CONFIG = {
    "architecture": "MONAI UNet (proposed channel depths, NO attention gates, NO residual units)",
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 2,
    "channels": "(32, 64, 128, 256, 512)  # matches proposed model",
    "strides": "(2, 2, 2, 2)",
    "num_res_units": 0,
    "kernel_size": 3,
    "up_kernel_size": 3,
    "dropout": 0.2,
}

DESCRIPTION = (
    "Proposed AttentionUnet stripped of its Attention Gates. The MONAI UNet "
    "keeps the proposed channel depths (32, 64, 128, 256, 512), strides, "
    "kernel sizes, and dropout. `num_res_units` is 0 to match AttentionUnet's "
    "block topology, so the only architectural difference vs. the proposed "
    "model is the absence of attention gates on the skip connections. "
    "Default learning rate is 1e-3 rather than 5e-3 (the proposed model's "
    "value) because the plain UNet lacks attention's regularisation cushion "
    "and diverges at the higher LR. Override with --lr 0.005 to reproduce "
    "the divergence for supplementary material."
)


if __name__ == "__main__":
    from ablation_utils import build_common_argparser, run_experiment_from_args

    parser = build_common_argparser(description=__doc__)
    # Override the inherited --lr default of 0.005 (which is right for the
    # attention-gated model but destabilises this one).
    parser.set_defaults(lr=0.001)
    args = parser.parse_args()
    run_experiment_from_args(
        args,
        experiment_tag="b_proposed_no_attention",
        experiment_description=DESCRIPTION,
        build_model=build_model,
        model_config_summary=MODEL_CONFIG,
    )
