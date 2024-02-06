import nibabel as nib
import numpy as np
import torch


def load_nifti(path):
    nii = nib.load(path)
    img = nii.get_fdata()
    img = np.asarry(img)

    return nii, img


def save_inference(data, path):
    ...


def niimg2tensor(img):
    return torch.from_numpy(img)


def add_rgb(img_tensor):
    img_tensor = torch.unsqueeze(img_tensor, dim=1)
    return torch.cat([img_tensor] * 3, dim=1)


def remove_color_channel(img_tensor):
    return torch.squeeze(img_tensor, dim=1)


def main():
    device = "gpu" if torch.cuda.is_available() else "cpu"

    unet = torch.hub.load(model="mazebusa...")
    unet = torch.load("path/to/saved/model")

    # TODO: right order?
    unet.to(device)
    unet.eval()

    nifti, img = load_nifti(...)

    img_tensor = niimg2tensor(img)

    img_tensor = add_rgb(img_tensor)

    img_inf = torch.zeros(img.shape)
    img_inf = torch.unsqueeze(img_inf, dim=1)

    with torch.set_grad_enabled(False):
        for i, x in enumerate(img_tensor):
            x = x.to(device)
            y_inf = unet(x)

            img_inf[i] = y_inf.detatch().cpu()

    img_inf = remove_color_channel(img_inf)
    img_inf = img_inf.numpy()

    inf_nifti = nib.nifti1.Nifti1Image(
        img_inf,
        nifti.affine,
        nifti.header,
    )

    nib.save(img_inf, "path/to/inferences/")
