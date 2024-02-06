import os.path as op
import random
import re
from glob import glob

import nibabel as nib
import numpy as np
import torch
import torchvision.transforms.v2 as tfms
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision.tv_tensors import Mask


class GreyToRGB(object):
    def __call__(self, volume_mask):
        volume, mask = volume_mask
        w, h, d = volume.shape
        volume += np.abs(np.min(volume))
        volume_max = np.max(volume)
        if volume_max > 0:
            volume /= volume_max
        ret = np.empty((w, h, d, 3), dtype=np.uint8)
        ret[:, :, :, 2] = ret[:, :, :, 1] = ret[:, :, :, 0] = volume * 255
        return ret, mask


class CropSample(object):
    def __call__(self, volume_mask):
        volume, mask = volume_mask

        def min_max_projection(axis: int):
            axis = [i for i in range(3) if i != axis]
            axis = tuple(axis)
            projection = np.max(volume, axis=axis)
            non_zero = np.nonzero(projection)
            return np.min(non_zero), np.max(non_zero) + 1

        z_min, z_max = min_max_projection(0)
        y_min, y_max = min_max_projection(1)
        x_min, x_max = min_max_projection(2)

        return (
            volume[z_min:z_max, y_min:y_max, x_min:x_max],
            mask[z_min:z_max, y_min:y_max, x_min:x_max],
        )


class PadSample(object):
    def __call__(self, volume_mask):
        volume, mask = volume_mask
        a = volume.shape[1]
        b = volume.shape[2]

        if a == b:
            return volume, mask
        diff = (max(a, b) - min(a, b)) / 2.0
        padding_insert = (
            int(np.floor(diff)),
            int(np.ceil(diff)),
        )
        if a > b:
            padding = ((0, 0), (0, 0), padding_insert)
        else:
            padding = ((0, 0), padding_insert, (0, 0))
        mask = np.pad(mask, padding, mode="constant", constant_values=0)
        padding = padding + ((0, 0),)
        volume = np.pad(volume, padding, mode="constant", constant_values=0)
        return volume, mask


class ResizeSample(object):
    def __init__(self, size=256):
        self.size = size

    def __call__(self, volume_mask):
        volume, mask = volume_mask
        v_shape = volume.shape
        out_shape = (v_shape[0], self.size, self.size)
        mask = resize(
            mask,
            output_shape=out_shape,
            order=0,
            mode="constant",
            cval=0,
            anti_aliasing=False,
        )
        volume = resize(
            volume,
            output_shape=out_shape,
            order=2,
            mode="constant",
            cval=0,
            anti_aliasing=False,
        )
        return volume, mask


class NormalizeVolume(object):
    def __call__(self, volume_mask):
        volume, mask = volume_mask
        p10 = np.percentile(volume, 10)
        p99 = np.percentile(volume, 99)
        volume = rescale_intensity(volume, in_range=(p10, p99))
        m = np.mean(volume, axis=(0, 1, 2))
        s = np.std(volume, axis=(0, 1, 2))
        volume = (volume - m) / s
        return volume, mask


class DataloaderImg(Dataset):
    """
    Dataset class for loading nifti files and masks for electrode extraction
    from a directory. The directory must have the following structure:

    root_dir
    ├── sub-01
    │   ├── ses-01
    │   │   ├── run-01
    │   │   │   ├── petra_.nii.gz
    │   │   │   ├── cylinder_plus_plug_ROI_FN.nii.gz
    │   │   ├── run-02
    │   │   │   ├── petra_.nii.gz
    │   │   │   ├── cylinder_plus_plug_ROI_FN.nii.gz
    ├── ses-02
    │   │   ├── run-01
    │   │   │   ├── petra_.nii.gz
    │   │   │   ├── cylinder_plus_plug_ROI_FN.nii.gz
    │   │   ├── run-02
    │   │   │   ├── petra_.nii.gz
    │   │   │   ├── cylinder_plus_plug_ROI_FN.nii.gz

    ...

    Parameters
    ----------
    root_dir : str
        The root directory of the dataset
    preprocessing : list[object], optional
        List of transforms to apply to the data, by default
        [CropSample(), PadSample(), ResizeSample(), NormalizeVolume()]
    transforms: list[object], optional
        List of transforms (standard torchvision.transforms)
        to apply to the data, by default
        None
    subset : str, optional
        The subset of the dataset to load, by default "train"
    random_sampling : bool, optional
        Whether to randomly sample the dataset, by default True
    validation_cases : int, optional
        The number of cases to use for validation, by default 10
    seed : int, optional
        The random seed to use for sampling, by default 42. Only used if random_sampling is True
        Should be the same for train and validation dataset.


    Returns
    -------
    volume : np.ndarray
        The volume of the MRI scan
    mask : np.ndarray
        The mask of the MRI scan

    """

    def __init__(
        self,
        root_dir: str,
        volume_suffix: str = "petra_cut_pads.nii.gz",
        mask_suffix: str = "cylinder_plus_plug_ROI.nii.gz",
        preprocessing: list[object] | None = None,
        transforms: list[object] | None = None,
        subset: str = "train",
        weighted_sampling: bool = True,
        all_slices: bool = False,
        validation_cases: int = 10,
        seed: int = 42,
    ):
        if subset not in ["train", "validation", "inference"]:
            raise ValueError("subset must be one of train, validation, or inference")

        if weighted_sampling == all_slices:
            raise ValueError("weighted_sampling and all_slices cannot be both True")

        self.root_dir = root_dir
        self.preprocessing = (
            tfms.Compose(preprocessing) if preprocessing is not None else None
        )
        self.transforms = tfms.Compose(transforms) if transforms is not None else None
        self.subject_pattern = op.join(
            self.root_dir,
            "sub-*",
            "electrode_extraction",
            "ses-*",
            "run-*",
        )
        volume = glob(op.join(self.subject_pattern, volume_suffix))
        volume.sort()
        if subset != "inference":
            masks = [op.join(op.dirname(i), mask_suffix) for i in volume]
            self.mask = [m for m in masks if op.exists(m)]
            self.volume = [v for v, m in zip(volume, masks) if op.exists(m)]
        else:
            masks = [None] * len(volume)

        if len(self.volume) != len(self.mask):
            raise ValueError(
                "Number of volumes and masks must be the same for training or validation"
            )

        if subset != "inference":
            np.random.seed(seed)
            subset_idx = np.random.choice(
                np.arange(len(self.volume)),
                size=validation_cases,
                replace=False,
            )
            validation_volumes = [
                i for idx, i in enumerate(self.volume) if idx in subset_idx
            ]
            validation_masks = [
                i for idx, i in enumerate(self.mask) if idx in subset_idx
            ]
            if subset == "validation":
                self.volume = validation_volumes
                self.mask = validation_masks
            else:
                self.volume = [i for i in self.volume if i not in validation_volumes]
                self.mask = [i for i in self.mask if i not in validation_masks]

        self.subset = subset

        self.weighted_sampling = weighted_sampling
        self.all_slices = all_slices

        sub_ses_run_idx = [
            "_".join(re.findall(r"(sub-[0-9]+|ses-[0-9]|run-[0-9]+)", x))
            for x in self.volume
        ]

        if subset == "training":
            np.savetxt(
                f"sub_ses_run_idx_{subset}.txt",
                sub_ses_run_idx,
            )

        self.sub_ses_run_idx = sub_ses_run_idx
        n_slices = nib.load(self.volume[0]).shape
        self.n_slices = n_slices[0]
        self.img_dim = n_slices[1:]
        self.sub_ses_run_slice_idx = [
            "_".join([i, str(j)])
            for i in self.sub_ses_run_idx
            for j in range(self.n_slices)
        ]

        self.val_slice = 0

    def __len__(self):
        return len(self.volume)

    def __getitem__(self, idx):
        slice_n = None

        if self.all_slices:
            sub_ses_run_slice = self.sub_ses_run_slice_idx[idx]
            sub_ses_run_idx, slice_n = sub_ses_run_slice.rsplit("_", 1)
            idx = self.sub_ses_run_idx.index(sub_ses_run_idx)

        if self.subset in ["validation", "inference"]:
            slice_n = self.val_slice

        volume_name = self.volume[idx]
        mask_name = self.mask[idx]
        volume = nib.load(volume_name)
        volume = np.asarray(volume.dataobj, dtype=np.float32)
        if self.subset != "inference":
            mask = nib.load(mask_name)
            mask = np.asarray(mask.dataobj, dtype=np.float32)
        else:
            mask = np.zeros_like(volume)

        if self.preprocessing is not None:
            volume, mask = self.preprocessing((volume, mask))

        if self.weighted_sampling and not self.subset in ["validation", "inference"]:
            slice_weights = mask.sum(axis=(1, 2))
            slice_weights = (
                slice_weights + (slice_weights.sum() * 0.1 / len(slice_weights))
            ) / (slice_weights.sum() * 1.1)

            slice_n = np.random.choice(np.arange(len(slice_weights)), p=slice_weights)

        volume = volume[slice_n]
        mask = mask[slice_n]

        volume = np.swapaxes(volume, 0, 2)
        mask = mask[..., np.newaxis]
        mask = np.swapaxes(mask, 0, 2)

        image_tensor = torch.from_numpy(volume)
        mask_tensor = Mask(torch.from_numpy(mask))

        if self.transforms is not None:
            image_tensor, mask_tensor = self.transforms(image_tensor, mask_tensor)

        return image_tensor, mask_tensor
