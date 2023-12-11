from skimage.transform import resize
from skimage.exposure import rescale_intensity
import numpy as np
import nibabel as nib
from glob import glob
import torchvision.transforms as tfms
import os.path as op
import re
import random
from torch.utils.data import Dataset

class GreyToRGB(object):
    
    def __call__(self, volume_mask):
        volume, mask = volume_mask
        w, h, d =  volume.shape
        volume += np.abs(np.min(volume))
        volume_max = np.max(volume)
        if volume_max > 0:
            volume /= volume_max 
        ret = np.empty((w, h, d, 3), dtype=np.uint8)
        ret[:, :, :, 2] = ret[:, :, :, 1] = ret[:, :, :, 0] = volume * 255
        return ret, mask
    
class ChannelSwitchBST(object):
    
    def __call__(self, volume_mask):
        volume, mask = volume_mask
        volume = np.transpose(volume, (0, 1, 3, 2))
        return volume, mask
    
class FixChannelDimension(object):
    
    def __call__(self, volume_mask):
        volume, mask = volume_mask
        volume = np.transpose(volume, (0, 1, 3, 2))
        return volume, mask

class CropSample(object):
    def __call__(self, volume_mask):
        volume, mask = volume_mask
        
        def min_max_projection(axis: int):
            
            axis = [i for i in range(4) if i != axis]
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
        out_shape = out_shape + (v_shape[3],)
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
    transforms : list[object], optional
        List of transforms to apply to the data, by default [CropSample(), PadSample(), ResizeSample(), NormalizeVolume()]
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
        transforms: list[object]
        | None = [
            GreyToRGB(),
            # ChannelSwitchBST(),
            CropSample(),
            PadSample(),
            ResizeSample(),
            NormalizeVolume(),
            # FixChannelDimension(),
        ],
        subset: str = "train",
        random_sampling: bool = True,
        validation_cases: int = 10,
        seed: int = 42,
    ):
        if subset not in ["train", "validation", "all"]:
            raise ValueError("subset must be one of train, validation, or all")

        self.root_dir = root_dir
        self.transforms = tfms.Compose(transforms)
        self.subject_pattern = op.join(
            self.root_dir,
            "sub-*",
            "electrode_extraction",
            "ses-*",
            "run-*",
        )
        self.volume = glob(op.join(self.subject_pattern, "petra_.nii.gz"))
        self.mask = glob(
            op.join(self.subject_pattern, "cylinder_plus_plug_ROI_FN.nii.gz")
        )  # TODO: change file name

        if not subset == "all":
            random.seed(seed)
            validation_volumes = random.sample(self.volume, k=validation_cases)
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
        # TODO: change so that it returns the actual number of files not number of subjects
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

        if self.transforms is not None:
            volume, mask = self.transforms((volume, mask))

        if self.random_sampling:
            slice_weights = mask.sum(axis=(1, 2))
            slice_weights = (
                slice_weights + (slice_weights.sum() * 0.1 / len(slice_weights))
            ) / (slice_weights.sum() * 1.1)

            slice_n = np.random.choice(np.arange(len(slice_weights)), p=slice_weights)

        volume = volume[slice_n]
        mask = mask[slice_n]

        image_tensor = torch.from_numpy(volume)
        mask_tensor = torch.from_numpy(mask)

        return image_tensor, mask_tensor

