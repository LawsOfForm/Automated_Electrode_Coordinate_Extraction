import os.path as op
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
import nibabel as nib
from torchvision.tv_tensors import Mask


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
        The random seed to use for sampling, by default 42. Only used
        if random_sampling is True. Should be the same for train and
        validation dataset.


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
        transforms: list[object] | None = None,
        subset: str = "train",
        validation_cases: int = 8,
        test_cases: int = 8,
        seed: int = 42,
    ):

        valid_subset = ["train", "validation", "test", "inference"]

        if subset not in valid_subset:
            raise ValueError("subset must be one of train, validation, or inference")

        self.root_dir = root_dir
        self.transforms = (
            tfms.Compose(transforms) if transforms is not None else None
        )  # !TODO: 3d Transforms
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
                "Number of volumes and masks must be the same for"
                "training or validation"
            )

        if subset != "inference":
            np.random.seed(seed)

            sampling_array = np.hstack(
                np.array(
                    np.repeat(
                        "train",
                        len(self.volume) - (validation_cases + test_cases),
                    )
                ),
                np.array(np.repeat("validation", validation_cases)),
                np.array(np.repeat("test", test_cases)),
            )
            np.random.shuffle(sampling_array)

            self.volume = self.volume[sampling_array == subset]
            self.mask = self.mask[sampling_array == subset]

        self.subset = subset

    def __len__(self):
        return len(self.volume)

    def __getitem__(self, idx):
        volume_name = self.volume[idx]
        mask_name = self.mask[idx]
        volume = nib.load(volume_name)
        volume = np.asarray(volume.dataobj, dtype=np.float32)

        if self.subset != "inference":
            mask = nib.load(mask_name)
            mask = np.asarray(mask.dataobj, dtype=np.float32)
        else:
            mask = np.zeros_like(volume)

        volume = np.swapaxes(volume, 0, 2)
        mask = mask[..., np.newaxis]
        mask = np.swapaxes(mask, 0, 2)

        image_tensor = torch.from_numpy(volume)
        mask_tensor = Mask(torch.from_numpy(mask))

        if self.transforms is not None:
            image_tensor, mask_tensor = self.transforms(image_tensor, mask_tensor)

        return image_tensor, mask_tensor

    def set_val_slice(self, val_slice):
        self.val_slice = val_slice
