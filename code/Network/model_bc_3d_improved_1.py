import os.path as op
from glob import glob
import matplotlib.pyplot as plt
import monai.transforms as tfms
import numpy as np
import torch
import torch.nn as nn
from monai.data import ArrayDataset, DataLoader, decollate_batch
from monai.losses import DiceFocalLoss, DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU, SurfaceDistanceMetric, ConfusionMatrixMetric
from monai.networks.nets import AttentionUnet
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from datetime import datetime
import os
from pathlib import Path

# define most import path variables
script_directory = Path(__file__).parent.resolve()
root = script_directory.parent.parent.resolve()

debug_dir_full = os.path.join(script_directory,"debug_images")
root_dataset = os.path.join('/media/MeMoSLAP_SUBJECTS/derivatives/automated_electrode_extraction')


# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir=os.path.join(script_directory,"runs/experiment_1"))

# Check if CUDA is available
print(torch.cuda.is_available())

def save_debug_batch(inputs, labels, predictions=None, debug_dir=debug_dir_full):
    """
    Save a slice of the input volume, ground truth mask, and optionally the model prediction.
    Only saves slices containing mask voxels or the slice with the maximum number of mask voxels.
    """
    os.makedirs(debug_dir, exist_ok=True)
    inputs = inputs.cpu().numpy()
    labels = labels.cpu().numpy()
    if predictions is not None:
        predictions = predictions.cpu().numpy()

    for i in range(min(3, inputs.shape[0])):  # Save up to 3 samples
        # Find slices with mask voxels
        mask_slices = np.sum(labels[i, 0], axis=(0, 1)) > 0
        if not np.any(mask_slices):
            continue  # Skip this sample if no mask voxels are found

        # Find the slice with the maximum number of mask voxels
        best_slice_idx = np.argmax(np.sum(labels[i, 0], axis=(0, 1)))

        fig, axes = plt.subplots(1, 3 if predictions is not None else 2, figsize=(15, 5))

        # Input volume slice
        axes[0].imshow(inputs[i, 0, :, :, best_slice_idx], cmap="gray")
        axes[0].set_title(f"Input Volume (Slice {best_slice_idx})")
        axes[0].axis("off")

        # Ground truth mask slice
        axes[1].imshow(labels[i, 0, :, :, best_slice_idx], cmap="jet", alpha=0.5)
        axes[1].set_title(f"Ground Truth Mask (Slice {best_slice_idx})")
        axes[1].axis("off")

        # Model prediction slice (if available)
        if predictions is not None:
            axes[2].imshow(predictions[i, 0, :, :, best_slice_idx], cmap="jet", alpha=0.5)
            axes[2].set_title(f"Model Prediction (Slice {best_slice_idx})")
            axes[2].axis("off")

        plt.tight_layout()

        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_batch_{timestamp}_sample_{i}_slice_{best_slice_idx}.png"
        filepath = os.path.join(debug_dir, filename)

        # Save the figure
        plt.savefig(filepath)
        plt.close(fig)

def subsetting(
    subset: str,
    vols: list[str],
    mask: list[str],
    validation_cases: int,
    test_cases: int,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """From a list of volume and mask files, create the specified subset."""
    vols = np.asarray(vols)
    mask = np.asarray(mask)
    np.random.seed(seed)

    train_cases = len(vols) - (validation_cases + test_cases)
    sampling_array = np.hstack([
        np.array(np.repeat("train", train_cases)),
        np.array(np.repeat("validation", validation_cases)),
        np.array(np.repeat("test", test_cases)),
    ])
    np.random.shuffle(sampling_array)

    return vols[sampling_array == subset], mask[sampling_array == subset]

def _extract_bids_key(path: str) -> str:
    """Extract sub_ses_run key using folder name for subject ID.

    Stops each entity at the next _ or / so acq-petra or folder suffixes
    do not bleed into the key.

    sub-006/unzipped/rsub-006_ses-4_acq-petra_run-01_PDw.nii  -> sub-006_ses-4_run-01
    sub-6114/unzipped/sub-6114_ses-1_run-02/mask.nii.gz        -> sub-6114_ses-1_run-02
    """
    import re as _re
    # Use the sub-* folder in the path — avoids grabbing rsub-* from the filename
    sub_m = _re.search(r'/(sub-[^/]+)/', path)
    sub   = sub_m.group(1) if sub_m else ""
    # Non-greedy: stop at next _ or /
    ses_m = _re.search(r'ses-([^_/]+)', path)
    ses   = f"ses-{ses_m.group(1)}" if ses_m else ""
    run_m = _re.search(r'run-([^_/]+)', path)
    run   = f"run-{run_m.group(1)}" if run_m else ""
    return "_".join(filter(None, [sub, ses, run]))


def _pair_volumes_and_masks(root: str):
    """Match each mask to the one volume that shares its sub/ses/run key.

    Only the 72 volumes that have a corresponding mask are returned.
    The 3369 - 72 volumes without masks are silently excluded — they have
    no supervision signal and should not be in the training set.
    """
    import re as _re
    volume_suffix = "rsub*.nii"
    mask_suffix   = "mask.nii.gz"

    all_volumes = sorted(glob(op.join(root, "sub-*", "unzipped", volume_suffix)))
    all_masks   = sorted(glob(op.join(root, "sub-*", "unzipped", "sub-*", mask_suffix)))

    # Build mask lookup: key -> mask path
    mask_lookup: dict = {}
    for m in all_masks:
        key = _extract_bids_key(m)
        if key not in mask_lookup:
            mask_lookup[key] = m
        else:
            print(f"WARNING: duplicate mask key '{key}', keeping first, skipping: {m}")

    # Only keep volumes that have a matching mask
    paired_vols, paired_masks = [], []
    for v in all_volumes:
        key = _extract_bids_key(v)
        if key in mask_lookup:
            paired_vols.append(v)
            paired_masks.append(mask_lookup[key])

    print(f"Using {len(paired_vols)} volumes that have masks (out of {len(all_volumes)} total volumes, {len(all_masks)} masks).")
    if len(paired_vols) == 0:
        # Print a sample of keys from each side to help debug
        print("Sample volume keys:", [_extract_bids_key(v) for v in all_volumes[:5]])
        print("Sample mask keys:",   [_extract_bids_key(m) for m in all_masks[:5]])
    return paired_vols, paired_masks


def create_dataset(
    root: str,
    subset: str,
    validation_cases: int,
    test_cases: int,
    seed: int = 42
) -> ArrayDataset:
    """Load Dataset and transformations for MONAI.

    Uses BIDS-entity-based pairing (not glob+sort) to correctly match volumes
    and masks even when subject IDs have inconsistent zero-padding.
    """
    volumes, masks = _pair_volumes_and_masks(root)

    if not volumes:
        raise ValueError(f"No matched volume/mask pairs found under: {root}")

    volumes, masks = subsetting(
        subset=subset,
        vols=volumes,
        mask=masks,
        validation_cases=validation_cases,
        test_cases=test_cases,
        seed=seed,
    )

    data_dicts = [{"image": v, "label": m} for v, m in zip(volumes, masks)]

    if subset == "train":
        transforms = tfms.Compose([
            # ---- Load & basic pre-processing ----
            tfms.LoadImaged(keys=["image", "label"]),
            tfms.EnsureChannelFirstd(keys=["image", "label"]),
            tfms.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            # ---- Synchronized spatial augmentation ----
            # Both keys receive the exact same random parameters:
            tfms.RandZoomd(keys=["image", "label"], prob=1.0,
                           min_zoom=0.7, max_zoom=1.3,
                           mode=("trilinear", "nearest")),
            tfms.RandRotated(keys=["image", "label"], prob=1.0,
                             range_x=0.5, range_y=0.5, range_z=0.5,
                             keep_size=True, mode=("bilinear", "nearest")),
            tfms.RandAffined(keys=["image", "label"], prob=1.0,
                             rotate_range=0.5, shear_range=0.3,
                             padding_mode="zeros",
                             mode=("bilinear", "nearest")),
            tfms.Resized(keys=["image", "label"],
                         spatial_size=(224, 288, 288),
                         mode=("trilinear", "nearest")),
            tfms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            tfms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            tfms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # ---- Intensity augmentation (image only) ----
            tfms.RandGaussianNoised(keys=["image"], prob=0.15, std=0.1),
            tfms.RandGaussianSmoothd(keys=["image"], prob=0.1,
                                     sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5)),
            tfms.RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.5, 2.0)),
            tfms.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.1),
        ])
    else:
        transforms = tfms.Compose([
            tfms.LoadImaged(keys=["image", "label"]),
            tfms.EnsureChannelFirstd(keys=["image", "label"]),
            tfms.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            tfms.Resized(keys=["image", "label"],
                         spatial_size=(224, 288, 288),
                         mode=("trilinear", "nearest")),
        ])

    from monai.data import Dataset
    return Dataset(data=data_dicts, transform=transforms)



class Network:
    def __init__(
        self,
        net,
        scaler,
        opt,
        loss_function,
        train_loader,
        val_loader,
        test_loader,
        dice_metric,
        hausdorff_metric,
        iou_metric,
        eval_num: int,
        max_iterations: int,
        root_dir: str,
        accumulation_steps: int = 8,
        early_stopping_patience: int = 15,
        grad_clip_norm: float = 1.0,
    ):
        self.net = net
        self.scaler = scaler
        self.opt = opt
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.dice_metric = dice_metric
        self.hausdorff_metric = hausdorff_metric
        self.iou_metric = iou_metric
        self.surface_metric = SurfaceDistanceMetric(include_background=False, symmetric=True)
        self.confusion_metric = ConfusionMatrixMetric(
            include_background=False,
            metric_name=["sensitivity", "precision"],
            reduction="mean",
            compute_sample=True,
        )
        self.eval_num = eval_num
        self.max_iterations = max_iterations
        self.epoch_loss_values = []
        self.metric_values = []
        self.dice_val_best = 0.0
        self.global_step_best = 0
        self.global_step = 0
        self.root_dir = root_dir
        self.accumulation_steps = accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0
        self.grad_clip_norm = grad_clip_norm
        self.scheduler = CosineAnnealingWarmRestarts(self.opt, T_0=1000, T_mult=1, eta_min=1e-6)

    def train(self):
        """Train-Loop for the UNet."""
        self.net.train()
        epoch_loss = 0.0
        step = 0
        epoch_iterator = tqdm(
            self.train_loader,
            desc="Training (X / X Steps) (loss=X.X)",
            dynamic_ncols=True,
        )

        for step, batch in enumerate(epoch_iterator, start=1):
            # Dict-transform batches: access by key
            x = batch["image"].cuda()
            y = batch["label"].cuda()

            with torch.amp.autocast('cuda'):
                logit_map = self.net(x)
                loss = self.loss_function(logit_map, y) / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if step % self.accumulation_steps == 0:
                # Gradient clipping prevents exploding gradients with small lesions
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_norm)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.scheduler.step()
                self.opt.zero_grad()

            epoch_loss += loss.item() * self.accumulation_steps

            if self.global_step % 300 == 0:
                with torch.no_grad():
                    pred = torch.argmax(logit_map.detach(), dim=1).unsqueeze(1)
                    save_debug_batch(x.detach(), y.detach(), pred)

            epoch_iterator.set_description(
                f"Training ({self.global_step} / {self.max_iterations} Steps) (loss={loss.item() * self.accumulation_steps:2.5f})"
            )

            writer.add_scalar('Loss/train', loss.item() * self.accumulation_steps, self.global_step)
            writer.add_scalar('Learning Rate', self.opt.param_groups[0]['lr'], self.global_step)

            if (self.global_step % self.eval_num == 0) or (self.global_step == self.max_iterations):
                dice_val, hausdorff_val, iou_val = self.validation()
                epoch_loss /= step
                self.epoch_loss_values.append(epoch_loss)
                self.metric_values.append(dice_val)

                writer.add_scalar('Dice/val', dice_val, self.global_step)
                writer.add_scalar('Hausdorff/val', hausdorff_val, self.global_step)
                writer.add_scalar('IoU/val', iou_val, self.global_step)

                if dice_val > self.dice_val_best:
                    self.dice_val_best = dice_val
                    self.global_step_best = self.global_step
                    torch.save(
                        self.net.state_dict(), op.join(self.root_dir, "best_metric_model.pth")
                    )
                    print(f"  ✓ Model saved! Best Dice: {self.dice_val_best:.4f}")
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        print(f"Early stopping triggered at step {self.global_step}.")
                        self.global_step = self.max_iterations  # signal outer loop to stop
                        return

            self.global_step += 1


    @torch.no_grad()
    def validation(self):
        """Validation of the UNet using sliding-window inference on full volumes."""
        post_pred = tfms.Compose([tfms.AsDiscrete(argmax=True, to_onehot=2)])
        post_label = tfms.Compose([tfms.AsDiscrete(to_onehot=2)])
        self.net.eval()
        dice_values, hausdorff_values, iou_values = [], [], []
        surface_values, sensitivity_values, precision_values = [], [], []

        for batch in self.val_loader:
            val_inputs = batch["image"].cuda()
            val_labels = batch["label"].cuda()

            # Sliding-window inference: evaluates full volumes without OOM
            val_output = sliding_window_inference(
                inputs=val_inputs,
                roi_size=(128, 128, 128),
                sw_batch_size=2,
                predictor=self.net,
                overlap=0.5,
            )

            val_output_ = [post_pred(i) for i in decollate_batch(val_output)]
            val_labels_ = [post_label(i) for i in decollate_batch(val_labels)]

            self.dice_metric(y_pred=val_output_, y=val_labels_)
            self.hausdorff_metric(y_pred=val_output_, y=val_labels_)
            self.iou_metric(y_pred=val_output_, y=val_labels_)
            self.surface_metric(y_pred=val_output_, y=val_labels_)
            self.confusion_metric(y_pred=val_output_, y=val_labels_)

            dice_values.append(self.dice_metric.aggregate().item())
            hausdorff_values.append(self.hausdorff_metric.aggregate().item())
            iou_values.append(self.iou_metric.aggregate().item())
            surface_values.append(self.surface_metric.aggregate().item())

            conf = self.confusion_metric.aggregate()
            sensitivity_values.append(conf[0].item())
            precision_values.append(conf[1].item())

            self.dice_metric.reset()
            self.hausdorff_metric.reset()
            self.iou_metric.reset()
            self.surface_metric.reset()
            self.confusion_metric.reset()

        d   = np.mean(dice_values)
        h   = np.mean(hausdorff_values)
        iou = np.mean(iou_values)
        asd = np.mean(surface_values)
        sen = np.mean(sensitivity_values)
        pre = np.mean(precision_values)
        f1  = 2 * (pre * sen) / (pre + sen + 1e-8)

        writer.add_scalar('Val/SurfaceDist', asd, self.global_step)
        writer.add_scalar('Val/Sensitivity', sen, self.global_step)
        writer.add_scalar('Val/Precision',   pre, self.global_step)
        writer.add_scalar('Val/F1',          f1,  self.global_step)

        print(
            f"  Val → Dice: {d:.4f}  IoU: {iou:.4f}  "
            f"Hausdorff: {h:.4f}  ASD: {asd:.4f}  "
            f"Sensitivity: {sen:.4f}  Precision: {pre:.4f}  F1: {f1:.4f}"
        )

        self.net.train()
        return d, h, iou

    @torch.no_grad()
    def test(self):
        """Test the UNet on the test dataset."""
        post_pred = tfms.Compose([tfms.AsDiscrete(argmax=True, to_onehot=2)])
        post_label = tfms.Compose([tfms.AsDiscrete(to_onehot=2)])
        self.net.eval()
        dice_values, hausdorff_values, iou_values = [], [], []
        surface_values, sensitivity_values, precision_values = [], [], []

        for batch in self.test_loader:
            test_inputs = batch["image"].cuda()
            test_labels = batch["label"].cuda()

            test_output = sliding_window_inference(
                inputs=test_inputs,
                roi_size=(128, 128, 128),
                sw_batch_size=2,
                predictor=self.net,
                overlap=0.5,
            )

            test_output_ = [post_pred(i) for i in decollate_batch(test_output)]
            test_labels_ = [post_label(i) for i in decollate_batch(test_labels)]

            self.dice_metric(y_pred=test_output_, y=test_labels_)
            self.hausdorff_metric(y_pred=test_output_, y=test_labels_)
            self.iou_metric(y_pred=test_output_, y=test_labels_)
            self.surface_metric(y_pred=test_output_, y=test_labels_)
            self.confusion_metric(y_pred=test_output_, y=test_labels_)

            dice_values.append(self.dice_metric.aggregate().item())
            hausdorff_values.append(self.hausdorff_metric.aggregate().item())
            iou_values.append(self.iou_metric.aggregate().item())
            surface_values.append(self.surface_metric.aggregate().item())
            conf = self.confusion_metric.aggregate()
            sensitivity_values.append(conf[0].item())
            precision_values.append(conf[1].item())

            self.dice_metric.reset()
            self.hausdorff_metric.reset()
            self.iou_metric.reset()
            self.surface_metric.reset()
            self.confusion_metric.reset()

        d   = np.mean(dice_values)
        h   = np.mean(hausdorff_values)
        iou = np.mean(iou_values)
        asd = np.mean(surface_values)
        sen = np.mean(sensitivity_values)
        pre = np.mean(precision_values)
        f1  = 2 * (pre * sen) / (pre + sen + 1e-8)

        writer.add_scalar('Dice/test',        d,   self.global_step)
        writer.add_scalar('Hausdorff/test',   h,   self.global_step)
        writer.add_scalar('IoU/test',         iou, self.global_step)
        writer.add_scalar('Test/SurfaceDist', asd, self.global_step)
        writer.add_scalar('Test/Sensitivity', sen, self.global_step)
        writer.add_scalar('Test/Precision',   pre, self.global_step)
        writer.add_scalar('Test/F1',          f1,  self.global_step)

        print(
            f"Test → Dice: {d:.4f}  IoU: {iou:.4f}  "
            f"Hausdorff: {h:.4f}  ASD: {asd:.4f}  "
            f"Sensitivity: {sen:.4f}  Precision: {pre:.4f}  F1: {f1:.4f}"
        )

def alt_main() -> None:
    vc, tc = 8, 8
    bs = 1
    seed = 1001
    accumulation_steps = 8
    early_stopping_patience = 15

    train_dataset = create_dataset(root=root_dataset, subset="train", validation_cases=vc, test_cases=tc, seed=seed)
    # num_workers=2 + pin_memory=False avoids the "received 0 items of ancdata"
    # crash that occurs when worker processes try to share pinned CUDA memory via
    # file descriptors and hit the OS ulimit (RLIMIT_NOFILE).
    # pin_memory=True with num_workers>2 on this system causes the pin_memory
    # thread to die silently, then the DataLoader hangs/crashes at validation.
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=2, pin_memory=False, shuffle=True)
    val_dataset = create_dataset(root=root_dataset, subset="validation", validation_cases=vc, test_cases=tc, seed=seed)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2, pin_memory=False)
    test_dataset = create_dataset(root=root_dataset, subset="test", validation_cases=vc, test_cases=tc, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, pin_memory=False)

    device = torch.device("cuda:0")

    # Same architecture as model.py — proven to work
    net = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2, 2),
        kernel_size=3,
        up_kernel_size=3,
        dropout=0.2,
    ).to(device)

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # AdamW: decoupled weight decay, more stable than plain Adam for small structures
    opt = AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    # DiceCELoss: numerically more stable than DiceFocalLoss for tiny structures.
    # ce_weight=[1,10] gives the foreground class 10× more weight in the CE term,
    # directly counteracting the extreme background/foreground imbalance of electrodes.
    ce_weight = torch.tensor([1.0, 10.0], device=device)
    loss_fn = DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        weight=ce_weight,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        lambda_dice=0.5,
        lambda_ce=0.5,
    )

    network = Network(
        net=net,
        scaler=torch.amp.GradScaler('cuda'),
        opt=opt,
        loss_function=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dice_metric=DiceMetric(reduction="mean", include_background=False, ignore_empty=True),
        hausdorff_metric=HausdorffDistanceMetric(include_background=False),
        iou_metric=MeanIoU(include_background=False, ignore_empty=True),
        eval_num=500,
        max_iterations=50_000,
        root_dir=op.dirname(op.abspath(__file__)),
        accumulation_steps=accumulation_steps,
        early_stopping_patience=early_stopping_patience,
        grad_clip_norm=1.0,
    )
    torch.cuda.empty_cache() # empty cache before each training step
    while network.global_step < network.max_iterations:
        network.train()

    print(f"Training completed. Best Dice: {network.dice_val_best:.4f} at iteration: {network.global_step_best}")

    # Load the best model and evaluate on the test dataset
    network.net.load_state_dict(torch.load(op.join(network.root_dir, "best_metric_model.pth")))
    network.test()

    writer.close()

if __name__ == "__main__":
    alt_main()