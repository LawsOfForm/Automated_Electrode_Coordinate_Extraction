"""
Ablation (d) — State-of-the-art comparator: SegResNet.

Why SegResNet (and not a Transformer)?
--------------------------------------
A thorough 2024-2025 literature review of 3D medical image segmentation
settles a question that's easy to get wrong:

  * Isensee et al., "nnU-Net Revisited", arXiv:2404.09556 (2024).
    Benchmarks CNN-, Transformer- and Mamba-based segmentation methods
    under matched compute and validation. Their conclusion: the recipe
    for SOTA in 3D medical segmentation is (1) CNN-based U-Net variants,
    especially ResNet and ConvNeXt encoders, (2) the nnU-Net framework,
    (3) scaling to modern hardware. Transformer claims of superiority
    were largely artifacts of weak baselines.

  * Bergner et al., "Benchmarking CNN-based Models against Transformer-
    based Models for Abdominal Multi-Organ Segmentation on the RATIC
    Dataset", arXiv:2603.18616 (2025). On 206 CT scans (small-to-medium,
    heterogeneous -- much like the present electrode dataset), the
    CNN-based **SegResNet** outperformed all hybrid transformer-based
    models (UNETR, SwinUNETR, UNETR++).

  * Myronenko, "3D MRI brain tumor segmentation using autoencoder
    regularization", arXiv:1810.11654 (2018) -- the SegResNet paper
    itself; winning solution of the BraTS 2018 challenge.

  * Tang et al., "Self-Supervised Pre-Training of Swin Transformers for
    3D Medical Image Analysis", CVPR 2022 -- SwinUNETR. On large-scale
    pre-trained CT it sets SOTA, but on small unpaired MRI datasets
    (without the 5,050-scan pre-training) it does NOT outperform a
    well-tuned CNN.

For an ablation against a ~66-subject electrode-segmentation task, the
honest "modern SOTA comparator" is a residual-encoder CNN U-Net, not a
Transformer. We pick SegResNet because (a) it has the strongest empirical
record on small datasets in the survey above, (b) it is the BraTS-winning
architecture, (c) it is implemented in MONAI so we can run it under the
identical pipeline as the other ablation variants.

The SegResNet configuration here:
    - 4 down/up stages with residual blocks at each
    - GroupNorm with 8 groups (better than BatchNorm for batch=1..2)
    - Dropout 0.2 in the bottleneck
    - Trilinear upsampling decoder
    - Same DiceFocalLoss + Adamax + cosine schedule as every other variant

What this does NOT include
--------------------------
The SegResNetVAE auto-encoder regularization branch from the original
paper. The VAE branch helps when training data is very small (~280 BraTS
cases) but doubles GPU memory; here it would force batch=1 and obscure
the architectural comparison. If you want to enable it later, replace
`SegResNet` with `SegResNetVAE` in `build_model()`.

GPU-aware batch sizing
----------------------
Same `--auto-tune-batch` knob as `c_model_channel_depth.py`. SegResNet has
~5M params (similar to the reduced AttentionUnet), so on an A100 we can
fit batch=4-8 comfortably.

Run
---
    python d_model_segresnet.py --auto-tune-batch
    python d_model_segresnet.py --batch-size 4
    python d_model_segresnet.py --init-filters 16  # smaller variant
    python d_model_segresnet.py --init-filters 32  # bigger variant
"""

from monai.networks.nets import SegResNet

from ablation_utils import build_common_argparser, run_experiment_from_args

# gpu_batch_helper is an OPTIONAL local module used only by --auto-tune-batch.
# It is not part of the shared pipeline and may not have been transferred to
# the cluster. Import it defensively so the script still runs (with a fixed
# batch size) if it's missing, instead of dying at import with ModuleNotFoundError.
try:
    from gpu_batch_helper import pick_batch_size, adjust_accumulation_for_batch
    _HAS_BATCH_HELPER = True
except ImportError:
    _HAS_BATCH_HELPER = False


def build_model_factory(init_filters, blocks_down, blocks_up, dropout):
    def _build():
        return SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            init_filters=init_filters,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            dropout_prob=dropout,
            norm=("GROUP", {"num_groups": 8}),
            upsample_mode="deconv",
        )
    return _build


def parse_args():
    p = build_common_argparser(description=__doc__)
    p.add_argument(
        "--init-filters", type=int, default=16,
        help="Base channel count of SegResNet. 16 is the BraTS default; "
             "raise to 24 or 32 for more capacity (memory permitting).",
    )
    p.add_argument(
        "--blocks-down", type=str, default="1,2,2,4",
        help="Comma-separated residual-block counts at each encoder stage. "
             "Default matches Myronenko (2018).",
    )
    p.add_argument(
        "--blocks-up", type=str, default="1,1,1",
        help="Comma-separated residual-block counts at each decoder stage. "
             "Default matches Myronenko (2018).",
    )
    p.add_argument(
        "--dropout", type=float, default=0.2,
        help="Dropout in the bottleneck and decoder.",
    )
    p.add_argument(
        "--auto-tune-batch", action="store_true",
        help="Pick the largest safe batch size for the current GPU.",
    )
    p.add_argument("--batch-cap", type=int, default=8)
    p.add_argument("--reserve-gb", type=float, default=4.0)
    return p.parse_args()


def main():
    args = parse_args()

    blocks_down = tuple(int(x) for x in args.blocks_down.split(","))
    blocks_up = tuple(int(x) for x in args.blocks_up.split(","))

    # SegResNet's memory footprint is dominated by `init_filters` and the
    # decoder stack -- we can re-use the channel-base heuristic from
    # gpu_batch_helper by passing the effective base width.
    effective_channels = (args.init_filters,)

    if args.auto_tune_batch:
        if _HAS_BATCH_HELPER:
            bs = pick_batch_size(effective_channels, reserve_gb=args.reserve_gb,
                                 cap=args.batch_cap)
            args.batch_size = bs
            args.accumulation_steps = adjust_accumulation_for_batch(
                args.accumulation_steps, bs, orig_effective_batch=8,
            )
            print(f"[d_segresnet] auto-tuned batch_size={args.batch_size}, "
                  f"accumulation_steps={args.accumulation_steps}")
        else:
            print("[d_segresnet] WARNING: --auto-tune-batch requested but "
                  "gpu_batch_helper is not importable; falling back to "
                  f"--batch-size {args.batch_size} "
                  f"--accumulation-steps {args.accumulation_steps}. "
                  "Transfer gpu_batch_helper.py or set batch size manually.")

    description = (
        "SegResNet (Myronenko 2018, BraTS-winning architecture). Chosen as a "
        "modern CNN-based SOTA comparator following the 2024 nnU-Net Revisited "
        "benchmark (Isensee et al.) and the 2025 RATIC small-dataset benchmark "
        "(Bergner et al.), both of which find that for ~tens-to-hundreds of "
        "subjects, residual-encoder CNNs outperform hybrid transformer models "
        "(UNETR, SwinUNETR, UNETR++). Trained under identical pipeline (loss, "
        "optimiser, scheduler, augmentations, data split) as the proposed "
        "Attention-UNet and the (a)/(b)/(c) ablation variants."
    )

    model_config = {
        "architecture": "MONAI SegResNet (Myronenko 2018; BraTS 2018 winner)",
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 2,
        "init_filters": args.init_filters,
        "blocks_down": str(blocks_down),
        "blocks_up": str(blocks_up),
        "dropout_prob": args.dropout,
        "norm": "GroupNorm(num_groups=8)",
        "upsample_mode": "deconv",
        "batch_size_runtime": args.batch_size,
        "accumulation_steps_runtime": args.accumulation_steps,
    }

    run_experiment_from_args(
        args,
        experiment_tag=f"d_segresnet_f{args.init_filters}",
        experiment_description=description,
        build_model=build_model_factory(
            args.init_filters, blocks_down, blocks_up, args.dropout,
        ),
        model_config_summary=model_config,
    )


if __name__ == "__main__":
    main()
