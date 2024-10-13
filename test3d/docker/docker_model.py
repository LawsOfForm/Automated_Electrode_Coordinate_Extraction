import logging
import os.path as op
from glob import glob
from typing import Any

import matplotlib.pyplot as plt
import monai.transforms as tfms
import numpy as np
import torch
from fileconfig import (INPUT_DIR, MASK_SUFFIX, OUTPUT_DIR, SUBJECT_PATTERN,
                        VOLUME_SUFFIX)
from monai.data import ArrayDataset, DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from tqdm import tqdm

RESIZE_DIM = (256, 256, 256)

logging.basicConfig(
    level=logging.INFO,
    filename=op.join(OUTPUT_DIR, "docker_model.log"),
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)


def debug_paths() -> None:
    """
    Some utility function that is used to debug the paths and files. Especially
    to check if docker is available to mount the volumes and masks, and write
    to the output directory.

    Returns
    ----------
    None
    """
    import os

    print(f"INPUT_DIR: {op.exists(INPUT_DIR)}")
    print(f"OUTPUT_DIR: {op.exists(OUTPUT_DIR)}")

    print(os.listdir(INPUT_DIR))

    vol_list = glob(op.join(INPUT_DIR, SUBJECT_PATTERN, VOLUME_SUFFIX))
    mask_list = glob(op.join(INPUT_DIR, SUBJECT_PATTERN, MASK_SUFFIX))
    print(f"Volumes: {vol_list}")
    print(f"Masks: {mask_list}")

    print(f"N volumes: {len(vol_list)}")
    print(f"N masks: {len(mask_list)}")


def check_paths() -> None:
    """
    Check if INPUT_DIR and OUTPUT_DIR exist. Otherwise, raise a ValueError.
    """
    for path in [INPUT_DIR, OUTPUT_DIR]:
        if op.exists(path):
            continue
        logging.error(f"Path {path} does not exist.")
        raise ValueError(f"Path {path} does not exist.")


def subsetting(
    subset: str,
    vols: list[str],
    mask: list[str],
    validation_cases: int,
    test_cases: int,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """From a list of volume and mask files create the subset specified

    Parameters
    -----------
    subset(str): Create a subset for "train", "validation", or "test"
        purposes
    vols(list[str]): List with paths to the image data
    mask(list[str]): List with paths to the ground truth mask data
    validation_cases(int): Number of Images for validation
    test_cases(int): Number of Images for tests
    seed(int): Seed for the randomization. Keep the same for the same
        model, default 42

    Returns
    -----------
    tuple[list[str],list[str]]: (image subset, mask subset)
    """
    vols = np.asarray(vols)
    mask = np.asarray(mask)

    np.random.seed(seed)

    train_cases = len(vols) - (validation_cases + test_cases)

    if train_cases <= 0:
        raise ValueError(f"Not enough data for training: {train_cases}")

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
    subset: str, validation_cases: int, test_cases: int, seed: int = 42
) -> ArrayDataset:
    """
    Load Dataset and transformations for MONAI

    Parameters
    -----------
    subset(str): Create a dataset for "train", "validation", or "test"
        purposes
    vols(list[str]): List with paths to the image data
    mask(list[str]): List with paths to the ground truth mask data
    validation_cases(int): Number of Images for validation
    test_cases(int): Number of Images for tests
    seed(int): Seed for the randomization. Keep the same for the same
        model, default 42

    Returns
    -----------
    ArrayDataset: Dataset for model

    Raises
    -----------
    ValueError: No MRI-volumes in the current path

    """
    volume_suffix: str = VOLUME_SUFFIX
    mask_suffix: str = MASK_SUFFIX
    root_dir = INPUT_DIR
    subject_pattern = op.join(root_dir, SUBJECT_PATTERN)

    volumes = glob(op.join(subject_pattern, volume_suffix))
    volumes.sort()
    masks = glob(op.join(subject_pattern, mask_suffix))
    masks.sort()

    if not volumes:
        msg = f"No MRI-volumes found at: {op.join(subject_pattern, volume_suffix)}"
        logging.error(msg)
        raise ValueError(msg)

    masks = [m for m in masks if op.exists(m)]
    volumes = [v for v, m in zip(volumes, masks) if op.exists(m)]

    if len(masks) != len(volumes):
        msg = "Number of masks and volumes do not match."
        logging.error(msg)
        raise ValueError(msg)

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
            tfms.RandZoom(1, min_zoom=0.7, max_zoom=1.3),
            tfms.RandRotate(
                prob=1,
                range_x=0.5,
                range_y=0.5,
                range_z=0.5,
                keep_size=True,
            ),
            tfms.RandAffine(
                prob=1, rotate_range=0.5, shear_range=0.5, padding_mode="zeros"
            ),
            tfms.Resize((RESIZE_DIM)),
            tfms.SignalFillEmpty(),
        ]
    )

    mask_tfms = tfms.Compose(
        [
            tfms.LoadImage(image_only=True),
            tfms.EnsureChannelFirst(),
            tfms.RandZoom(1, min_zoom=0.7, max_zoom=1.3),
            tfms.RandRotate(
                prob=1,
                range_x=0.5,
                range_y=0.5,
                range_z=0.5,
                keep_size=True,
            ),
            tfms.RandAffine(
                prob=1, rotate_range=0.5, shear_range=0.5, padding_mode="zeros"
            ),
            tfms.Resize((RESIZE_DIM)),
            tfms.SignalFillEmpty(),
        ]
    )

    return ArrayDataset(volumes, vol_tfms, masks, mask_tfms)


class Network:
    """
    A class used to represent the neural network training and validation process.

    Attributes
    ----------
    net : torch.nn.Module
        The neural network model to be trained and validated.
    scaler : torch.cuda.amp.GradScaler
        The gradient scaler for mixed precision training.
    opt : torch.optim.Optimizer
        The optimizer used for training the model.
    loss_function : torch.nn.Module
        The loss function used to compute the training loss.
    train_loader : torch.utils.data.DataLoader
        The DataLoader for the training dataset.
    val_loader : torch.utils.data.DataLoader
        The DataLoader for the validation dataset.
    dice_metric : monai.metrics.DiceMetric
        The metric used to evaluate the model performance.
    eval_num : int
        The number of iterations between each evaluation.
    max_iterations : int
        The maximum number of training iterations.
    epoch_loss_values : list
        A list to store the loss values for each epoch.
    metric_values : list
        A list to store the metric values for each evaluation.
    dice_val_best : float
        The best Dice metric value achieved during training.
    global_step_best : int
        The global step at which the best Dice metric value was achieved.
    global_step : int
        The current global step of the training process.
    root_dir : str
        The root directory for input data.

    Methods
    -------
    train():
        Executes the training loop for the neural network.
    validation():
        Executes the validation loop for the neural network.
    """

    batch_idx = {"image": 0, "mask": 1}

    def __init__(
        self,
        net,
        scaler,
        opt,
        loss_function,
        train_loader,
        val_loader,
        dice_metric,
        eval_num: int,
        max_iterations: int,
        root_dir: str,
    ):
        self.net = net
        self.scaler = scaler
        self.opt = opt
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dice_metric = dice_metric
        self.eval_num = eval_num
        self.max_iterations = max_iterations
        self.epoch_loss_values = []
        self.metric_values = []
        self.dice_val_best = 0
        self.global_step_best = 0
        self.global_step = 0
        self.root_dir = root_dir

    def train(self):
        """
        Train-Loop for the UNet
        """
        self.net.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(
            self.train_loader,
            desc="Training (X / X Steps) (loss=X.X)",
            dynamic_ncols=True,
        )
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (
                batch[Network.batch_idx["image"]].cuda(),
                batch[Network.batch_idx["mask"]].cuda(),
            )

            with torch.cuda.amp.autocast():
                logit_map = self.net(x)
                loss = self.loss_function(logit_map, y)

            with open(op.join(OUTPUT_DIR, "loss.txt"), "a") as f:
                f.write(f"{str(loss.item())}\n")

            self.opt.zero_grad()
            self.scaler.scale(loss).backward()
            epoch_loss += loss.item()
            self.scaler.unscale_(self.opt)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()
            epoch_iterator.set_description(
                f"Training ({self.global_step} / {self.max_iterations} Steps) (loss={loss:2.5f})"
            )

            if (
                (self.global_step % self.eval_num != 0)
                and (self.global_step != self.max_iterations)
                and (self.global_step != 1)
            ):
                self.global_step += 1
                continue
            dice_val = self.validation()
            epoch_loss /= step
            with open(op.join(OUTPUT_DIR, "epoch_loss.txt"), "a") as f:
                f.write(f"{str(loss.item())}\n")
            self.epoch_loss_values.append(epoch_loss)
            self.metric_values.append(dice_val)
            if dice_val < self.dice_val_best:
                logging.info(
                    "Model Was Not Saved! Current Best Avg. Dice: "
                    f"{self.dice_val_best} Current Avg. Dice: {dice_val}"
                )
                self.global_step += 1
                return

            self.dice_val_best = dice_val
            self.global_step_best = self.global_step
            torch.save(
                self.net.state_dict(),
                op.join(OUTPUT_DIR, "best_metric_model.pth"),
            )
            logging.info(
                f"Model Was Saved! Current Best Avg. Dice: {self.dice_val_best}"
                f" Current Avg. Dice: {dice_val}"
            )
            self.global_step += 1

    @torch.no_grad()
    def validation(self):
        """
        Validation of the UNet
        """
        post_pred = tfms.Compose([tfms.AsDiscrete(argmax=True, to_onehot=2)])
        post_label = tfms.Compose([tfms.AsDiscrete(to_onehot=2)])
        epoch_iterator_val = tqdm(
            self.val_loader,
            desc="Validate (X / X Steps) (dice=X.X)",
            dynamic_ncols=True,
        )
        self.net.eval()
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (
                batch[Network.batch_idx["image"]].cuda(),
                batch[Network.batch_idx["mask"]].cuda(),
            )
            val_output = self.net(val_inputs)

            self.dice_metric(
                y_pred=[post_pred(i) for i in decollate_batch(val_output)],
                y=[post_label(i) for i in decollate_batch(val_labels)],
            )
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (self.global_step, 10.0)
            )

        mean_dice_val = self.dice_metric.aggregate().item()
        with open(op.join(OUTPUT_DIR, "mean_dice_val.txt"), "a") as f:
            f.write(f"{str(mean_dice_val)}\n")
        self.dice_metric.reset()
        return mean_dice_val


def plot_training(network: Network) -> Any:
    _, axes = plt.subplots(nrows=1, ncols=2)

    for data, ax in zip(
        [network.epoch_loss_values, network.metric_values],
        axes.flatten(),
    ):
        x = [network.eval_num * (i + 1) for i in range(len(data))]
        ax.plot(x, data)
        plt.xlabel("Iteration")

    return axes


def main() -> None:
    check_paths()

    print(f"Cuda device: {torch.cuda.get_device_name(0)}")

    vc, tc = 2, 2
    bs = 8
    seed = 1001
    train_dataset = create_dataset(
        subset="train", validation_cases=vc, test_cases=tc, seed=seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
    )

    val_dataset = create_dataset(
        subset="validation", validation_cases=vc, test_cases=tc, seed=seed
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda")
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        act="RELU",
    ).to(device)

    network = Network(
        net=net,
        scaler=torch.cuda.amp.GradScaler(),
        opt=torch.optim.Adam(net.parameters(), lr=1e-3),
        # opt=torch.optim.SGD(
        #    net.parameters(),
        #    lr=5e-3,
        #    momentum=0.2
        # ),
        loss_function=DiceLoss(to_onehot_y=True, softmax=True),
        train_loader=train_loader,
        val_loader=val_loader,
        dice_metric=DiceMetric(reduction="mean", include_background=False),
        eval_num=500,
        max_iterations=30_000,
        root_dir=INPUT_DIR,
    )

    while network.global_step < network.max_iterations:
        network.train()

    logging.info(
        f"train completed, best_metric: {network.dice_val_best:.4f}"
        f" at iteration: {network.global_step_best}"
    )

    ax = plot_training(network)
    plt.save(op.join(OUTPUT_DIR, "training_plot.png"), ax)


if __name__ == "__main__":
    main()
    # debug_paths()
