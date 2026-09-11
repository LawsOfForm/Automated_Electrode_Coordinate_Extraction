import torch
import sys
from pathlib import Path
from collections import defaultdict


def infer_monai_unet_params(state: dict) -> dict:
    """Infer MONAI UNet architecture parameters from weight tensor shapes."""
    info = {}

    # Input channels: first conv weight shape = (out_ch, in_ch, kD, kH, kW)
    first_key = next(
        (k for k in state if "conv.weight" in k and not "upconv" in k
         and not "attention" in k and not "merge" in k),
        None
    )
    if first_key:
        shape = tuple(state[first_key].shape)
        info["spatial_dims"] = len(shape) - 2  # e.g. 3 for 3D
        info["in_channels"] = shape[1]

    # Output channels: final conv weight shape = (out_ch, in_ch, ...)
    last_key = next(
        (k for k in reversed(list(state.keys())) if "conv.weight" in k),
        None
    )
    if last_key:
        info["out_channels"] = tuple(state[last_key].shape)[0]

    # Kernel size from first conv
    if first_key:
        shape = tuple(state[first_key].shape)
        kernels = shape[2:]
        info["kernel_size"] = kernels[0] if len(set(kernels)) == 1 else kernels

    # Feature channels per encoder level: collect all unique out_channels from
    # encoder conv blocks (keys matching model.0, model.1.submodule.0, etc.)
    feature_map = {}
    for k, v in state.items():
        if "conv.weight" in k and "upconv" not in k and "attention" not in k \
                and "merge" not in k and "model.2" not in k:
            # depth = number of "submodule" occurrences in key
            depth = k.count("submodule")
            out_ch = tuple(v.shape)[0]
            if depth not in feature_map:
                feature_map[depth] = out_ch
    if feature_map:
        levels = sorted(feature_map.keys())
        info["channels"] = tuple(feature_map[l] for l in levels)
        info["num_levels (strides)"] = len(levels)

    # Stride: inferred from upconv weight shape (in_ch = out_ch * stride^dims if transposed)
    upconv_keys = [k for k in state if "upconv.up.conv.weight" in k]
    if upconv_keys:
        shape = tuple(state[upconv_keys[0]].shape)
        # upconv weight: (in_ch, out_ch, kD, kH, kW) for ConvTranspose
        info["upsample_kernel"] = shape[2:]

    # Attention: presence of attention gates
    info["attention"] = any("attention" in k for k in state)

    # Normalization: presence of batch norm running stats
    info["normalization"] = "batch" if any(
        "running_mean" in k for k in state) else "unknown"

    # Activation: presence of PReLU weight (learnable)
    if any("adn.A.weight" in k for k in state):
        info["activation"] = "PReLU (learnable)"
    else:
        info["activation"] = "unknown (no learnable activation weights found)"

    # Dropout: no weights stored for dropout, cannot infer rate
    info["dropout"] = "unknown (not stored in weights)"

    return info


def inspect_checkpoint(path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    print(f"\n{'='*60}")
    print(f"  Checkpoint: {Path(path).name}")
    print(f"{'='*60}")

    # ── Case 1: raw state_dict ────────────────────────────────────
    is_raw = not isinstance(ckpt, dict) or all(
        isinstance(v, torch.Tensor) for v in ckpt.values()
    )

    if is_raw:
        state = ckpt if isinstance(ckpt, dict) else ckpt
        print("\n[Type] Raw state_dict — only weights were saved.")
        print("       Loss values, metrics, and hyperparameters (LR, batch size,")
        print("       epochs, loss function) are NOT stored in this file.\n")

        # Architecture inference
        print("[Inferred Architecture Parameters]")
        params = infer_monai_unet_params(state)
        for k, v in params.items():
            print(f"  {k:<35s}: {v}")

        # Parameter count
        total = sum(v.numel() for v in state.values() if isinstance(v, torch.Tensor))
        trainable = sum(
            v.numel() for k, v in state.items()
            if isinstance(v, torch.Tensor) and "running" not in k
            and "num_batches" not in k
        )
        print(f"\n[Parameter Count]")
        print(f"  Total tensors              : {len(state)}")
        print(f"  Total parameters (all)     : {total:,}")
        print(f"  Trainable parameters (est) : {trainable:,}")

        # Layer summary
        print(f"\n[Layer Summary]")
        for k, v in state.items():
            print(f"  {k:<60s}  {str(tuple(v.shape))}")

        print(f"\n[What Cannot Be Recovered From This File]")
        print("  - Learning rate / optimizer type (e.g. Adamax)")
        print("  - Batch size")
        print("  - Number of training epochs")
        print("  - Loss function (e.g. DiceLoss, DiceCELoss)")
        print("  - Validation metrics / Dice scores per epoch")
        print("  - Dropout rate")
        print("  - Data augmentation settings")
        print("  → These must be retrieved from the original training script.")
        return

    # ── Case 2: full checkpoint dict ─────────────────────────────
    print(f"\n[Type] Full checkpoint dict")
    print(f"[Keys] {list(ckpt.keys())}\n")

    meta_keys = ["epoch", "step", "loss", "val_loss", "train_loss",
                 "metric", "lr", "learning_rate",
                 "best_metric", "best_metric_epoch", "dice", "val_dice"]
    print("[Training Metadata]")
    found = False
    for k in meta_keys:
        if k in ckpt:
            print(f"  {k}: {ckpt[k]}")
            found = True
    if not found:
        print("  (none found)")

    config_keys = ["config", "hyper_parameters", "hparams",
                   "hyperparameters", "args", "params", "cfg"]
    print("\n[Config / Hyperparameters]")
    found = False
    for k in config_keys:
        if k in ckpt:
            cfg = ckpt[k]
            print(f"  [{k}]")
            if isinstance(cfg, dict):
                for ck, cv in cfg.items():
                    print(f"    {ck}: {cv}")
            else:
                print(f"    {cfg}")
            found = True
    if not found:
        print("  (none found — not saved by training script)")

    loss_keys = ["loss_function", "criterion", "loss_fn", "loss_name"]
    print("\n[Loss Function]")
    found = False
    for k in loss_keys:
        if k in ckpt:
            print(f"  {k}: {ckpt[k]}")
            found = True
    if not found:
        print("  (not saved in this checkpoint)")

    if "optimizer" in ckpt or "optimizer_state_dict" in ckpt:
        opt = ckpt.get("optimizer", ckpt.get("optimizer_state_dict"))
        print("\n[Optimizer State]")
        if isinstance(opt, dict) and "param_groups" in opt:
            for i, pg in enumerate(opt["param_groups"]):
                pg_display = {k: v for k, v in pg.items() if k != "params"}
                print(f"  param_group[{i}]: {pg_display}")
        else:
            print("  (present but unreadable format)")

    if "scheduler" in ckpt or "lr_scheduler" in ckpt:
        sched = ckpt.get("scheduler", ckpt.get("lr_scheduler"))
        print("\n[LR Scheduler]")
        if isinstance(sched, dict):
            for k, v in sched.items():
                if not isinstance(v, (list, torch.Tensor)):
                    print(f"  {k}: {v}")

    state_key = next(
        (k for k in ["model", "state_dict", "model_state_dict", "net"] if k in ckpt),
        None
    )
    if state_key:
        state = ckpt[state_key]
        print(f"\n[Inferred Architecture Parameters]  (from '{state_key}')")
        for k, v in infer_monai_unet_params(state).items():
            print(f"  {k:<35s}: {v}")

        total = sum(v.numel() for v in state.values() if isinstance(v, torch.Tensor))
        print(f"\n[Parameter Count]")
        print(f"  Total parameters: {total:,}")

        print(f"\n[Layer Summary]")
        for k, v in state.items():
            print(f"  {k:<60s}  {str(tuple(v.shape))}")

    known = set(meta_keys + config_keys + loss_keys +
                ["optimizer", "optimizer_state_dict", "scheduler", "lr_scheduler",
                 "model", "state_dict", "model_state_dict", "net"])
    extra = {k: ckpt[k] for k in ckpt if k not in known}
    if extra:
        print("\n[Other Keys]")
        for k, v in extra.items():
            print(f"  {k}: {type(v).__name__} = {str(v)[:120]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <path/to/model.pth>")
        sys.exit(1)
    inspect_checkpoint(sys.argv[1])
