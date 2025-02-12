import os.path as op
from glob import glob

import matplotlib.pyplot as plt
import monai.transforms as tfms
import numpy as np
import torch, gc
from monai.data import ArrayDataset, DataLoader, decollate_batch
from monai.losses import DiceLoss, DiceFocalLoss, GeneralizedDiceFocalLoss, TverskyLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet, AttentionUnet
from tqdm import tqdm

"""
class WeightedLoss(monai.losses.DiceLoss):
#    def __init__(self, alpha=0.3, beta=0.3, gamma=0.7, delta=0.7, to_onehot_y=True, softmax=True, include_background=True, reduction = "mean", batch=True, **kwargs):
    def __init__(self, alpha=0.3, beta=0.3, gamma=0.7, delta=0.7, to_onehot_y=True, softmax=True, include_background=True,  batch=True, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.include_background = include_background
        self.reduction = reduction
        self.batch = batch
        self.dice_loss = DiceLoss(**kwargs)

    def forward(self, y_pred, y_true):
        # Convert inputs to float tensors
        y_pred = y_pred.float()

        # Apply softmax to predictions if specified
        if self.softmax:
            y_pred = torch.softmax(y_pred, dim=1)

        # Convert true labels to one-hot encoding if specified
        if self.to_onehot_y:
            y_true = one_hot(y_true, num_classes=y_pred.shape[1], include_background=self.include_background)

        # Calculate components
        fp = self.alpha * torch.sum(y_pred * (1 - y_true), dim=self.batch_dim(y_pred)) # false positive
        fn = self.beta * torch.sum((1 - y_pred) * y_true, dim=self.batch_dim(y_pred)) # false negative
        tp = self.gamma * torch.sum((1 - y_pred) * (1 - y_true), dim=self.batch_dim(y_pred)) # correct negative
        tn = self.delta * torch.sum(y_pred * y_true, dim=self.batch_dim(y_pred)) # correct positiv

        # Calculate Dice loss
        dice_loss = self.dice_loss(y_pred, y_true)

        # Apply reduction if specified
        #if self.reduction == "mean":
        fp = fp.mean()
        fn = fn.mean()
        tp = tp.mean()
        tn = tn.mean()
        dice_loss = dice_loss.mean()

        return fp + fn + tp + tn + dice_loss

    def batch_dim(self, tensor):
        if self.batch:
            return (0,)
        else:
            return ()
"""
# Example usage
#criterion = WeightedLoss(alpha=0.7, beta=0.3, gamma=0.5, delta=0.5, to_onehot_y=True, softmax=True, include_background=True, #reduction='mean', batch=True)
#y_pred = torch.tensor([[[[0.1, 0.9], [0.6, 0.4]], [[0.3, 0.7], [0.8, 0.2]]]])
#y_true = torch.tensor([[[0, 1], [1, 0]]])
#loss = criterion(y_pred, y_true)
#print(loss)

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
    volume_suffix: str = "petra_cut_*.nii.gz"
    mask_suffix: str = "cylinder_plus_*.nii.gz"
    #mask_suffix: str = "cylinder_ROI.nii.gz"
    #root_dir = "./data"
    #root_dir = "automated_electrode_extraction"
    #subject_pattern = op.join(root_dir, "heads_true")

    #automated_electrode_extraction"
    root_dir= "~/data/heads_true"
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
            tfms.Resize((256, 256, 256)),
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
            tfms.Resize((256, 256, 256)),
            tfms.SignalFillEmpty(),
        ]
    )

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
            x, y = batch[0].cuda(), batch[1].cuda()

            with torch.cuda.amp.autocast():
                logit_map = self.net(x)
                loss = self.loss_function(logit_map, y)

            with open("loss.txt", "a") as f:
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
            with open("epoch_loss.txt", "a") as f:
                f.write(f"{str(loss.item())}\n")
            self.epoch_loss_values.append(epoch_loss)
            self.metric_values.append(dice_val)
            if dice_val < self.dice_val_best:
                print(
                    "Model Was Not Saved! Current Best Avg. Dice: "
                    f"{self.dice_val_best} Current Avg. Dice: {dice_val}"
                )
                self.global_step += 1
                return

            self.dice_val_best = dice_val
            self.global_step_best = self.global_step
            torch.save(
                self.net.state_dict(), op.join(self.root_dir, "best_metric_model.pth")
            )
            print(
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
        # post_pred2 = tfms.Compose([tfms.AsDiscrete(argmax=True)])
        # sigmoid = torch.nn.Sigmoid()
        epoch_iterator_val = tqdm(
            self.val_loader,
            desc="Validate (X / X Steps) (dice=X.X)",
            dynamic_ncols=True,
        )
        self.net.eval()
        for batch in epoch_iterator_val:
            val_inputs, val_labels = batch[0].cuda(), batch[1].cuda()
            val_output = self.net(val_inputs)

            val_output_ = val_output.cpu().numpy()
            val_labels_ = val_labels.cpu().numpy()

            np.save("val_out2.npy", val_output_)
            np.save("val_in2.npy", val_labels_)

            self.dice_metric(
                y_pred=[post_pred(i) for i in decollate_batch(val_output)],
                y=[post_label(i) for i in decollate_batch(val_labels)],
            )
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (self.global_step, 10.0)
            )
        mean_dice_val = self.dice_metric.aggregate().item()
        with open("mean_dice_val.txt", "a") as f:
            f.write(f"{str(mean_dice_val)}\n")
        self.dice_metric.reset()
        return mean_dice_val


def alt_main() -> None:
    vc, tc = 8, 8
    #bs = 32    # for complete 3d images choose smaller batch sizes
    bs = 6
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
        val_dataset, batch_size=bs, num_workers=2, pin_memory=torch.cuda.is_available()
    )
    device = torch.device("cuda:0")
    #net = UNet(
    #    spatial_dims=3,
    #    in_channels=1,
    #    out_channels=2,
    #    channels=(32, 64, 128, 256, 512),
    #    strides=(2, 2, 2, 2),
    #    num_res_units=2,
    #    act="RELU",
    #    act="PRELU",
    #).to(device)
    
    net = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        kernel_size = 3,
        up_kernel_size = 3
    ).to(device)
    

    network = Network(
        net=net,
        scaler=torch.cuda.amp.GradScaler(), # default 1e-3
        #https://docs.monai.io/en/1.3.0/optimizers.html#learningratefinder
        opt=torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4), # default 1e-4
        #opt = torch.optim.ASGD(net.parameters(), lr=1e-4, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0.001,),
        #opt=torch.optim.SGD(
        #    net.parameters(),
        #    lr=1e-4,
        #    momentum=0.9
        # ),
        #loss_function=DiceLoss(to_onehot_y=True, softmax=True), #default SR
        
        #loss_function = DiceLoss(sigmoid=True)
        #loss_function = GeneralizedDiceFocalLoss(sigmoid=True)
        #lambda_dice, labda focal L_unified_focal = λ * L_dice + (1 - λ) * L_focal  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8785124/
    
    # include_background = False: When the non-background segmentations are small compared to the total image size. Using include_background=False in a two-class scenario can underestimate the actual loss, as the Dice loss may not properly capture the importance of the smaller foreground regions. 1  (https://github.com/Project-MONAI/MONAI/issues/2509)
            
        #loss_function = DiceFocalLoss(include_background = False, lambda_dice=0.4, lambda_focal=0.6, reduction = 'mean',to_onehot_y=True, softmax=True),
        loss_function = TverskyLoss(include_background = True, alpha=0.2, beta=0.8,reduction = 'mean',to_onehot_y=True, softmax=True, batch = True),
        #loss_fcuntion = WeightedLoss(alpha=0.3, beta=0.3, gamma=0.7, delta=0.7, to_onehot_y=True, softmax=True, include_background=True, batch=True)
        # alpha = false positiv, beta = false negativ, gamma = correct positive, delta = correct negativ
        
#loss = criterion(y_pred, y_true)
        
        #loss_function = DiceFocalLoss(include_background = True, lambda_dice=0.4, lambda_focal=0.6,to_onehot_y=True, softmax=True),
        #dice_metric = DiceMetric(reduction="mean"),
        train_loader=train_loader,
        val_loader=val_loader,
        dice_metric=DiceMetric(reduction="mean", include_background=True),
        #dice_metric=DiceMetric(include_background=False),
        eval_num=500,
        max_iterations=30_000,
        root_dir=op.dirname(op.abspath(__file__)),
    )

    while network.global_step < network.max_iterations:
        network.train()

    print(
        f"train completed, best_metric: {network.dice_val_best:.4f}"
        f" at iteration: {network.global_step_best}"
    )

    _, axes = plt.subplots(nrows=1, ncols=2)

    for data, ax in zip(
        [network.epoch_loss_values, network.metric_values],
        axes.flatten(),
    ):
        x = [network.eval_num * (i + 1) for i in range(len(data))]
        ax.plot(x, data)
        plt.xlabel("Iteration")

    plt.show()
