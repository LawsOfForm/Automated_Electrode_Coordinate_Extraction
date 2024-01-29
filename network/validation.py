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
