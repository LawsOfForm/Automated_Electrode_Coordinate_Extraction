import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm


def load_nifti(path):
    nii = nib.load(path)
    img = nii.get_fdata()
    img = np.asarray(img)

    return nii, img


def niimg2tensor(img):
    return torch.from_numpy(img)


def add_rgb(img_tensor):
    img_tensor = torch.unsqueeze(img_tensor, dim=1)
    return torch.cat([img_tensor] * 3, dim=1)


def remove_color_channel(img_tensor):
    return torch.squeeze(img_tensor, dim=1)


def main():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    unet = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=False,
    )
    unet = torch.load("unet_epochs_1000_batchsize_15.pt")

    unet.eval()
    unet.to(device)

    nifti, img = load_nifti(
        "/media/MeMoSLAP_Subjects/SUBJECTS_XNAT/sub-4012/ses-1/anat/sub-4012_ses-1_acq-petra_run-01_PDw.nii.gz"
    )  # TODO: path as input

    img_tensor = niimg2tensor(img)

    img_tensor = add_rgb(img_tensor)

    img_inf = torch.zeros(img.shape)
    img_inf = torch.unsqueeze(img_inf, dim=1)

    with torch.set_grad_enabled(False):
        for i, x in tqdm(enumerate(img_tensor)):
            x = torch.unsqueeze(x, dim=0)
            x = x.to(device, dtype=torch.float32)
            y_inf = unet(x)

            img_inf[i] = y_inf.detach().cpu()

    img_inf = remove_color_channel(img_inf)
    img_inf = img_inf.numpy()
    img_inf = np.round(img_inf)

    inf_nifti = nib.nifti1.Nifti1Image(
        img_inf,
        nifti.affine,
        nifti.header,
    )

    nib.save(inf_nifti, "inference.nii.gz")  # TODO: sensible default


if __name__ == "__main__":
    main()
