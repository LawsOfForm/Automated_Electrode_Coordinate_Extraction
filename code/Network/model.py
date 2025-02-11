import os.path as op
from glob import glob
import matplotlib.pyplot as plt
import monai.transforms as tfms
import numpy as np
import torch
from monai.data import ArrayDataset, DataLoader, decollate_batch
from monai.losses import DiceLoss, DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
from monai.networks.nets import AttentionUnet, UNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from datetime import datetime
import os
from pathlib import Path
#import optuna

# define most import path variables
script_directory = Path(__file__).parent.resolve()
root = script_directory.parent.parent.resolve()

debug_dir_full = os.path.join(script_directory,"debug_images")
root_dataset = os.path.join(root,'dataset/media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction')


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

def check_matching_substrings(volumes, masks):
    for vol, mask in zip(volumes, masks):
        vol_parts = op.basename(vol).split('_')
        vol_substring = '_'.join([part for part in vol_parts if part.startswith(('sub-', 'ses-', 'run-'))])
        mask_parts = mask.split('unzipped/')
        if len(mask_parts) > 1:
            mask_folder = mask_parts[1].split('/')[0]
            if vol_substring not in mask_folder:
                print(f"No match for:\nVolume: {vol}\nMask: {mask}\n")
        else:
            print(f"Invalid mask path format: {mask}\n")

def create_dataset(
    root: str,
    subset: str, 
    validation_cases: int, 
    test_cases: int, 
    seed: int = 42
) -> ArrayDataset:
    """Load Dataset and transformations for MONAI."""
    volume_suffix: str = "rsub*.nii"
    mask_suffix: str = "mask.nii.gz"

    subject_pattern_vol = op.join(root, "sub-*", "unzipped")
    subject_pattern_mask = op.join(root, "sub-*", "unzipped", "sub-*")

    volumes = glob(op.join(subject_pattern_vol, volume_suffix))
    volumes.sort()
    masks = glob(op.join(subject_pattern_mask, mask_suffix))
    masks.sort()

    if len(volumes)!=len(masks):
        print('Check if every volume has one mask')
    check_matching_substrings(volumes, masks) #only needet for dataset because of different naming scheeme

    if not volumes:
        raise ValueError("No MRI-volumes found.")

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

        # Debug: Check the shape of the loaded volumes and masks
    for vol, mask in zip(volumes, masks):
        vol_data = tfms.LoadImage(image_only=True)(vol)
        mask_data = tfms.LoadImage(image_only=True)(mask)
        print(f"Volume shape: {vol_data.shape}, Mask shape: {mask_data.shape}")  # Should be 3D


    vol_tfms = tfms.Compose([
        tfms.LoadImage(image_only=True),
        tfms.ScaleIntensity(),
        tfms.EnsureChannelFirst(),
        tfms.RandZoom(1, min_zoom=0.7, max_zoom=1.3),
        tfms.RandRotate(prob=1, range_x=0.5, range_y=0.5, range_z=0.5, keep_size=True),
        tfms.RandAffine(prob=1, rotate_range=0.5, shear_range=0.5, padding_mode="zeros"),
        #tfms.Resize((256, 256, 256)),
        tfms.Resize((224,288, 288)), # original rsub*.nii.gz size
        tfms.RandFlip(prob=0.5, spatial_axis=0),
        tfms.RandGaussianNoise(prob=0.1, mean=0.0, std=0.1),
        tfms.RandGaussianSmooth(prob=0.1, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5)),
        tfms.RandAdjustContrast(prob=0.1, gamma=(0.5, 2.0)),
        #tfms.RandSpatialCrop(roi_size=(128, 128, 128), random_size=False),
        tfms.RandShiftIntensity(offsets=0.1, prob=0.01),
        tfms.RandCoarseDropout(holes=10, spatial_size=5, fill_value=None, prob=0.01),
        #tfms.SignalFillEmpty(),
    ])

    mask_tfms = tfms.Compose([
        tfms.LoadImage(image_only=True),
        tfms.EnsureChannelFirst(),
        tfms.RandZoom(1, min_zoom=0.7, max_zoom=1.3, mode = 'nearest'),
        tfms.RandRotate(prob=1, range_x=0.5, range_y=0.5, range_z=0.5, keep_size=True, mode = 'nearest'),
        tfms.RandAffine(prob=1, rotate_range=0.5, shear_range=0.5, padding_mode="zeros", mode = 'nearest'),
        #tfms.Resize((256, 256, 256)),
        tfms.Resize((224, 288, 288)),  # original rsub*.nii.gz size
        tfms.RandFlip(prob=0.5, spatial_axis=0),
        #tfms.RandSpatialCrop(roi_size=(128, 128, 128), random_size=False),
        tfms.RandShiftIntensity(offsets=0.1, prob=0.01),
        tfms.RandCoarseDropout(holes=10, spatial_size=5, fill_value=None, prob=0.01),
        #tfms.SignalFillEmpty(),
    ])

    return ArrayDataset(volumes, vol_tfms, masks, mask_tfms)

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
        accumulation_steps: int = 4,  # Gradient accumulation steps
        #early_stopping_patience: int=5, # Early stoppng patience
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.2,
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
        self.eval_num = eval_num
        self.max_iterations = max_iterations
        self.epoch_loss_values = []
        self.metric_values = []
        self.dice_val_best = 0
        self.global_step_best = 0
        self.global_step = 0
        self.root_dir = root_dir
        self.accumulation_steps = accumulation_steps
        #self.early_stopping_patience = early_stopping_patience
        #self.early_stopping_counter = 0  # Counter for early stopping
        self.learning_rate = learning_rate
        self.weight_decay=weight_decay
        self.dropout=dropout
        # Learning Rate Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(self.opt, T_0=1000, T_mult=1, eta_min=1e-5)

    def train(self):
        """Train-Loop for the UNet."""
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
            x, y = batch[0].cuda(), batch[1].cuda()

            #with torch.cuda.amp.autocast():
            with torch.amp.autocast('cuda'):
                logit_map = self.net(x)
                loss = self.loss_function(logit_map, y) / self.accumulation_steps  # Normalize loss

            self.scaler.scale(loss).backward()

            # Gradient accumulation: Update weights every `accumulation_steps` batches
            if (step + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.opt)
                self.scaler.update()
                self.scheduler.step()  # Update learning rate
                self.opt.zero_grad()

            epoch_loss += loss.item() * self.accumulation_steps  # Scale loss back up
            
            # Debugging: Save visualizations every N steps
            if self.global_step % 300 == 0:  # Adjust frequency as needed
                with torch.no_grad():
                    pred = torch.argmax(logit_map.detach(), dim=1).unsqueeze(1)  # Convert logits to class predictions
                    save_debug_batch(x.detach(), y.detach(), pred)

            epoch_iterator.set_description(
                f"Training ({self.global_step} / {self.max_iterations} Steps) (loss={loss.item() * self.accumulation_steps:2.5f})"
            )

            # Log training loss and learning rate to TensorBoard
            writer.add_scalar('Loss/train', loss.item() * self.accumulation_steps, self.global_step)
            writer.add_scalar('Learning Rate', self.opt.param_groups[0]['lr'], self.global_step)

            if (self.global_step % self.eval_num == 0) or (self.global_step == self.max_iterations):
                dice_val, hausdorff_val, iou_val = self.validation()
                epoch_loss /= step
                self.epoch_loss_values.append(epoch_loss)
                self.metric_values.append(dice_val)

                # Log validation metrics to TensorBoard
                writer.add_scalar('Dice/val', dice_val, self.global_step)
                writer.add_scalar('Hausdorff/val', hausdorff_val, self.global_step)
                writer.add_scalar('IoU/val', iou_val, self.global_step)

                # Early stopping logic
                if dice_val > self.dice_val_best:
                    self.dice_val_best = dice_val
                    self.global_step_best = self.global_step
                    torch.save(
                        self.net.state_dict(), op.join(self.root_dir, "best_metric_model.pth")
                    )
                    print(f"Model Saved! Best Dice: {self.dice_val_best:.4f}")
                #    self.early_stopping_counter = 0  # Reset counter
                #else:
                #    self.early_stopping_counter += 1
                #    if self.early_stopping_counter >= self.early_stopping_patience:
                #        print(f"Early stopping triggered at step {self.global_step}.")
                #        break

            self.global_step += 1


    @torch.no_grad()
    def validation(self):
        """Validation of the UNet."""
        post_pred = tfms.Compose([tfms.AsDiscrete(argmax=True, to_onehot=2)])
        post_label = tfms.Compose([tfms.AsDiscrete(to_onehot=2)])
        self.net.eval()
        dice_values, hausdorff_values, iou_values = [], [], []

        for batch in self.val_loader:
            val_inputs, val_labels = batch[0].cuda(), batch[1].cuda()
            val_output = self.net(val_inputs)

            # Debug: Check the shape of the model's output
            print(f"Model output shape: {val_output.shape}")  # Should be (batch_size, channels, depth, height, width)


            val_output_ = [post_pred(i) for i in decollate_batch(val_output)]
            val_labels_ = [post_label(i) for i in decollate_batch(val_labels)]

            self.dice_metric(y_pred=val_output_, y=val_labels_)
            self.hausdorff_metric(y_pred=val_output_, y=val_labels_)
            self.iou_metric(y_pred=val_output_, y=val_labels_)

            dice_values.append(self.dice_metric.aggregate().item())
            hausdorff_values.append(self.hausdorff_metric.aggregate().item())
            iou_values.append(self.iou_metric.aggregate().item())

            self.dice_metric.reset()
            self.hausdorff_metric.reset()
            self.iou_metric.reset()

            # Log validation metrics to TensorBoard
        #writer.add_scalar('Dice/val', np.mean(dice_values), self.global_step)
        #writer.add_scalar('Hausdorff/val', np.mean(hausdorff_values), self.global_step)
        #writer.add_scalar('IoU/val', np.mean(iou_values), self.global_step)

        print(f"Test Results - Dice: {np.mean(dice_values):.4f}, Hausdorff: {np.mean(hausdorff_values):.4f}, IoU: {np.mean(iou_values):.4f}")


        return np.mean(dice_values), np.mean(hausdorff_values), np.mean(iou_values)

    @torch.no_grad()
    def test(self):
        """Test the UNet on the test dataset."""
        post_pred = tfms.Compose([tfms.AsDiscrete(argmax=True, to_onehot=2)])
        post_label = tfms.Compose([tfms.AsDiscrete(to_onehot=2)])
        self.net.eval()
        dice_values, hausdorff_values, iou_values = [], [], []

        for batch in self.test_loader:
            test_inputs, test_labels = batch[0].cuda(), batch[1].cuda()
            test_output = self.net(test_inputs)

            test_output_ = [post_pred(i) for i in decollate_batch(test_output)]
            test_labels_ = [post_label(i) for i in decollate_batch(test_labels)]

            self.dice_metric(y_pred=test_output_, y=test_labels_)
            self.hausdorff_metric(y_pred=test_output_, y=test_labels_)
            self.iou_metric(y_pred=test_output_, y=test_labels_)

            dice_values.append(self.dice_metric.aggregate().item())
            hausdorff_values.append(self.hausdorff_metric.aggregate().item())
            iou_values.append(self.iou_metric.aggregate().item())

            self.dice_metric.reset()
            self.hausdorff_metric.reset()
            self.iou_metric.reset()

        # Log test metrics to TensorBoard
        writer.add_scalar('Dice/test', np.mean(dice_values), self.global_step)
        writer.add_scalar('Hausdorff/test', np.mean(hausdorff_values), self.global_step)
        writer.add_scalar('IoU/test', np.mean(iou_values), self.global_step)

        print(f"Test Results - Dice: {np.mean(dice_values):.4f}, Hausdorff: {np.mean(hausdorff_values):.4f}, IoU: {np.mean(iou_values):.4f}")

def alt_main() -> None:
    
    vc, tc = 8, 8
    bs = 1  # Batch size
    seed = 1001
    accumulation_steps = 8  # Gradient accumulation steps
    #early_stopping_patience = 5  # Early stopping patience

    train_dataset = create_dataset(root = root_dataset, subset="train", validation_cases=vc, test_cases=tc, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=4, pin_memory=torch.cuda.is_available(), shuffle=True)
    val_dataset = create_dataset(root = root_dataset, subset="validation", validation_cases=vc, test_cases=tc, seed=seed)
    val_loader = DataLoader(val_dataset, batch_size=bs, num_workers=4, pin_memory=torch.cuda.is_available())
    test_dataset = create_dataset(root = root_dataset, subset="test", validation_cases=vc, test_cases=tc, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=bs, num_workers=4, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda:0")
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

    background_value=False
    network = Network(
        net=net,
        scaler=torch.amp.GradScaler('cuda'),
        opt=torch.optim.Adam(net.parameters()),
        #opt=torch.optim.Adamax(net.parameters(),lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0), # default net.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        loss_function=DiceFocalLoss(
            include_background=background_value, 
            lambda_dice=0.3, 
            lambda_focal=0.7, 
            to_onehot_y=True, 
            softmax=True, 
            gamma=2.5, 
            smooth_nr = 1e-6, 
            smooth_dr = 1e-6
            ),
        
        #loss_function=GeneralizedFocalDiceLoss(to_onehot_y=True, softmax=True),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dice_metric=DiceMetric(reduction="mean", include_background=background_value, ignore_empty=True), #default include_backlground=True but for small segments not recommended
        hausdorff_metric=HausdorffDistanceMetric(include_background=background_value), #default include_backlground=True but for small segments not recommended
        iou_metric=MeanIoU(include_background=background_value, ignore_empty=True),
        eval_num=500,
        max_iterations=50_000,
        root_dir=op.dirname(op.abspath(__file__)),
        accumulation_steps=accumulation_steps,
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