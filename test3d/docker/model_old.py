import argparse
import os.path as op
from glob import glob

import matplotlib.pyplot as plt
import monai.transforms as tfms
import numpy as np
import torch
from monai.data import ArrayDataset, DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from tqdm import tqdm


def subsetting(
    subset: str,
    vols: list,
    mask: list,
    validation_cases: int,
    test_cases: int,
    seed: int = 42,
):
    vols = np.asarray(vols)
    mask = np.asarray(mask)

    np.random.seed(seed)

    train_cases = len(vols) - (validation_cases + test_cases)
    sampling_array = np.hstack(
        [
            np.array(np.repeat("train", train_cases)),
            np.array(np.repeat("validation", validation_cases)),
            np.array(np.repeat("test", test_cases)),
        ]
    )
    np.random.shuffle(sampling_array)

    return vols[sampling_array == subset], mask[sampling_array == subset]


def create_dataset(
    root_dir: str,
    subset: str,
    validation_cases: int,
    test_cases: int,
    seed: int = 42,
):
    volume_suffix: str = "petra_cut_pads.nii.gz"
    mask_suffix: str = "cylinder_plus_plug_ROI.nii.gz"
    subject_pattern = op.join(
        root_dir,
        "sub-*",
        "electrode_extraction",
        "ses-*",
        "run-*",
    )

    volumes = glob(op.join(subject_pattern, volume_suffix))
    volumes.sort()
    masks = glob(op.join(subject_pattern, mask_suffix))
    masks.sort()

    masks = [m for m in masks if op.exists(m)]
    volumes = [v for v, m in zip(volumes, masks) if op.exists(m)]

    volumes, masks = subsetting(
        subset=subset,
        vols=volumes,
        mask=masks,
        validation_cases=validation_cases,
        test_cases=test_cases,
        seed=seed,
    )

    vol_tfms = tfms.Compose(
        [
            tfms.LoadImage(image_only=True),
            tfms.ScaleIntensity(),
            tfms.EnsureChannelFirst(),
            tfms.RandRotate(prob=1, keep_size=True),
            tfms.RandZoom(1, min_zoom=0.7, max_zoom=1.3),
            tfms.RandAffine(1),
        ]
    )

    mask_tfms = tfms.Compose(
        [
            tfms.LoadImage(image_only=True),
            tfms.EnsureChannelFirst(),
            tfms.RandRotate(prob=0.7, keep_size=True),
            tfms.RandZoom(1, min_zoom=0.7, max_zoom=1.3),
            tfms.RandAffine(1),
        ]
    )

    return ArrayDataset(volumes, vol_tfms, masks, mask_tfms)


@torch.no_grad()
def validation(net, global_step, epoch_iterator_val, dice_metric):
    net.eval()
    for batch in epoch_iterator_val:
        val_inputs, val_labels = batch[0].cuda(), batch[1].cuda()
        val_output = net(val_inputs)
        dice_metric(y_pred=val_output, y=val_labels)
        epoch_iterator_val.set_description(
            f"Validate ({global_step} / {10.0} Steps)"
        )  # noqa: B038
    mean_dice_val = dice_metric.aggregate().item()
    dice_metric.reset()
    return mean_dice_val


def train(
    net,
    loss_function,
    global_step,
    train_loader,
    val_loader,
    dice_val_best,
    global_step_best,
    opt,
    dice_metric,
    eval_num,
    max_iterations,
    epoch_loss_values,
    metric_values,
    root_dir,
):
    net.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = batch[0].cuda(), batch[1].cuda()
        with torch.cuda.amp.autocast():
            logit_map = net(x)
            loss = loss_function(logit_map, y)

        epoch_loss += loss.item()
        loss.backward()
        opt.step()
        epoch_iterator.set_description(  # noqa: B038
            f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})"
        )

        if (global_step % eval_num != 0) or (global_step != max_iterations):
            global_step += 1
            continue

        epoch_iterator_val = tqdm(
            val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
        )
        dice_val = validation(net, global_step, epoch_iterator_val, dice_metric)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        metric_values.append(dice_val)
        if dice_val < dice_val_best:
            dice_val_best = dice_val
            global_step_best = global_step
            torch.save(net.state_dict(), op.join(root_dir, "best_metric_model.pth"))
            print(
                f"Model Was Saved! Current Best Avg. Dice: {dice_val_best}"
                f" Current Avg. Dice: {dice_val}"
            )
        else:
            print(
                f"Model Was Not Saved! Current Best Avg. Dice: {dice_val_best}"
                f" Current Avg. Dice: {dice_val}"
            )
        global_step += 1
    return (
        net,
        loss_function,
        global_step,
        dice_val_best,
        global_step_best,
        opt,
        eval_num,
        epoch_loss_values,
        metric_values,
    )


def args_input():
    """
    Input for the docker file

    Parameter
    -----------
    None

    Returns
    ----------
    Namespace: Including path to root_dir and out_dir
    """
    parser = argparse.ArgumentParser(prog="3D U-Net", description="Create a 3D U-Net")

    parser.add_argument(
        "-r",
        "--root_dir",
        help="Path to data as in our local machines, i.e., directory containing `sub-*` files.",
    )

    parser.add_argument(
        "-o", "--out_path", help="Directory to which all outputs are written."
    )

    return parser.parse_args()


def main():
    args = args_input()
    vc, tc = 8, 8
    train_dataset = create_dataset(
        root_dir=args.root_dir, subset="train", validation_cases=vc, test_cases=tc
    )
    train_loader = DataLoader(
        train_dataset, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available()
    )
    val_dataset = create_dataset(
        root_dir=args.root_dir, subset="validation", validation_cases=vc, test_cases=tc
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda:0")
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_function = DiceLoss(sigmoid=True)
    dice_metric = DiceMetric(reduction="mean")
    lr = 1e-3
    opt = torch.optim.Adam(net.parameters(), lr)
    max_iterations = 30_000
    eval_num = 500
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []

    root_dir = args.out_path
    while global_step < max_iterations:
        (
            net,
            loss_function,
            global_step,
            dice_val_best,
            global_step_best,
            opt,
            eval_num,
            epoch_loss_values,
            metric_values,
        ) = train(
            net,
            loss_function,
            global_step,
            train_loader,
            val_loader,
            dice_val_best,
            global_step_best,
            opt,
            dice_metric,
            eval_num,
            max_iterations,
            epoch_loss_values,
            metric_values,
            root_dir,
        )
    net.load_state_dict(torch.load(op.join(root_dir, "best_metric_model.pth")))

    print(
        f"train completed, best_metric: {dice_val_best:.4f}"
        f" at iteration: {global_step_best}"
    )

    _, axes = plt.subplot(nrows=1, ncols=2)

    for data, title, ax in zip(
        [epoch_loss_values, metric_values],
        ["Average Loss per Iteration", "Validation Mean Dice"],
        axes.ravel(),
    ):
        ax.title(title)
        x = [eval_num * (i + 1) for i in range(len(data))]
        ax.xlabel("Iteration")
        ax.plot(x, data)
        plt.title(title)

    plt.savefig(op.join(root_dir, "loss_metric.png"), dpi=300)


if __name__ == "__main__":
    main()
