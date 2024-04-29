import torch
from loss import DiceLoss
from tqdm import tqdm
import numpy as np


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


def generalized_dsc(pred, mask, thres_clip=1e-7, round_pred=False):
    pred_cat = np.concatenate([pred, 1 - pred], axis=1)
    if round_pred:
        pred_cat = np.round(pred_cat)
    mask_cat = np.concatenate([mask, 1 - mask], axis=1)

    w_l = mask_cat.sum(axis=(2, 3))
    w_l = 1 / np.clip(w_l * w_l, a_min=thres_clip, a_max=None)

    intersect = (pred_cat * mask_cat).sum(axis=(2, 3))
    intersect = intersect * w_l

    demoninator = (pred_cat + mask_cat).sum(axis=(2, 3))
    demoninator = np.clip(demoninator * w_l, a_min=thres_clip, a_max=None)

    dsc = 2 * intersect.sum(axis=1) / demoninator.sum(axis=1)

    return dsc


def accuracy(y_pred, y_true):
    y_pred = y_pred.round()
    return np.sum(y_pred == y_true, axis=(1, 2, 3)) / np.prod(y_pred.shape[1:])
