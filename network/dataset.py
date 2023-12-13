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
    custom_transforms : list[object], optional
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
        custom_transforms: list[object]
        | None = None,
        transforms: list[object] | None=None,
        subset: str = "train",
        random_sampling: bool = True,
        validation_cases: int = 10,
        seed: int = 42,
    ):
        if subset not in ["train", "validation", "all"]:
            raise ValueError("subset must be one of train, validation, or all")

        self.root_dir = root_dir
        self.custom_transforms = tfms.Compose(custom_transforms) if custom_transforms is not None else None
        self.transforms = tfms.Compose(transforms) if transforms is not None else None
        self.subject_pattern = op.join(
            self.root_dir,
            "sub-*",
            "electrode_extraction",
            "ses-*",
            "run-*",
        )
        self.subject_pattern
        self.volume = glob(op.join(self.subject_pattern, "petra_.nii.gz"))
        self.volume.sort()
        self.mask = glob(
            op.join(self.subject_pattern, "cylinder_plus_plug_ROI_FN.nii.gz")
        )  # TODO: change file name
        self.mask.sort()
        
        if len(self.volume) != len(self.mask):
            raise ValueError("Number of volumes and masks must be the same")
      
        if not subset == "all":
            random.seed(seed)
            validation_volumes = random.sample(self.volume, k=validation_cases)
            # TODO: also sample masks
            if subset == "validation":
                self.volume = validation_volumes
            else:
                self.volume = sorted(
                    list(set(self.volume).difference(validation_volumes))
                )

        self.sub_ses_run_idx = [
            "_".join(re.findall(r"(sub-[0-9]+|ses-[0-9]|run-[0-9]+)", x))
            for x in self.volume
        ]
        print(self.sub_ses_run_idx)

        self.random_sampling = random_sampling

        # TODO: create a sub index

    def __len__(self):
        return len(self.volume)

    def __getitem__(self, idx):
        volume_name = self.volume[idx]
        mask_name = self.mask[idx]
        volume = nib.load(volume_name)
        # change to numpy
        volume = np.asarray(volume.dataobj, dtype=np.float32)
        # change to PIL

        print(f"volume Size: {volume.size}")

        mask = nib.load(mask_name)
        # change to numpy
        mask = np.asarray(mask.dataobj, dtype=np.float32)
        # change to PIL
        # mask = Image.fromarray(mask.astype("uint8"), "L")

        print(f"mask Size: {mask.size}")

        if self.custom_transforms is not None:
            volume, mask = self.custom_transforms((volume, mask))

        if self.random_sampling:
            slice_weights = mask.sum(axis=(1, 2))
            slice_weights = (
                slice_weights
                + (slice_weights.sum() * 0.1 / len(slice_weights))
            ) / (slice_weights.sum() * 1.1)

            slice_n = np.random.choice(
                np.arange(len(slice_weights)), p=slice_weights
            )

        volume = volume[slice_n]
        mask = mask[slice_n]
        
        volume = np.swapaxes(volume, 0, 2)
        mask = mask[..., np.newaxis]
        mask = np.swapaxes(mask, 0, 2)

        image_tensor = torch.from_numpy(
            volume
        )  # might be torch.squeeze(torch.from_numpy(volume, 0))
        mask_tensor = torch.from_numpy(mask)
        
        if self.transforms is not None:
            image_tensor, mask_tensor = self.transforms(image_tensor, mask_tensor)

        return image_tensor, mask_tensor
