import torch
from loss import DiceLoss
from tqdm import tqdm


def main(
    unet,
    val_loader,
    device,
):
    unet.eval()

    dsc_loss = DiceLoss()

    validation_pred = []
    validation_true = []

    for i, data in enumerate(val_loader):
        x, y_true = data
        x, y_true = x.to(device), y_true.to(device)

        loss = dsc_loss(y_pred, y_true)

        # add loss to tensorboard

        ...


def get_validation_loop_len(validation_dataset, bs):
    n_slices = validation_dataset.n_slices
    val_slice = validation_dataset.val_slice
    n_subs = len(validation_dataset)

    if n_subs % bs != 0:
        print(
            f"Warning: validation dataset size ({n_subs}) is not divisible by batch size ({bs})."
            "There might be more efficient ways to handle this."
        )

    return (n_slices - val_slice) * (n_subs // bs or 1)
