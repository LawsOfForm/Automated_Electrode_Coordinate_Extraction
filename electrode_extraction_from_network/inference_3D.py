#use with linux meinzer2, only loaded dataset
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms as T
import os.path as op
#from paths import root_dir
import glob
from monai.data import decollate_batch
from monai.transforms import (
    AsDiscreted,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from monai.networks.nets import UNet, AttentionUnet

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    #unet = torch.hub.load(
    #    "mateuszbuda/brain-segmentation-pytorch",
    #    "unet",
    #    in_channels=3,
    #    out_channels=1,
    #    init_features=32,
    #    pretrained=False,
    #)
    #unet = torch.load(op.join(op.dirname(__file__), "unet_epochs_1000_batchsize_15.pt"))

    unet = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        kernel_size = 3,
        up_kernel_size = 3
    ).to(device)
    #unet = UNet(
    #    spatial_dims=3,
    #    in_channels=1,
    #    out_channels=2,
    #    #channels=(32, 64, 128, 256, 512),
    #    channels=(32, 64, 128, 256, 512),
    #    strides=(2, 2, 2, 2),
    #    num_res_units=2,
    #    act="RELU",
    #)


    path_unet = "/media/Data03/Studies/MeMoSLAP/code/Automated_Electrode_Coordinate_Extraction/electrode_extraction_from_network/Network"
    unet.load_state_dict(torch.load(op.join(path_unet, "best_metric_model_attentionUnet_ADAM_tversky_backround_True.pth"), map_location='cpu'))
    unet.eval()
    unet.to(device)

    # nifti, img = load_nifti(
    #     "/media/MeMoSLAP_Subjects/SUBJECTS_XNAT/sub-4012/ses-1/anat/sub-4012_ses-1_acq-petra_run-01_PDw.nii.gz"
    # )  # TODO: path as input
    #base_dir = root_dir
    nifti_path = "/media/Data03/Thesis/Dabelstein/derivatives/"
    sub_dirs = glob.glob(f'{nifti_path}**/petra_cut_pads.nii.gz', recursive=True)

    for sub_dir in sub_dirs:

        img = None
        nifti = None
        img_tensor = None
        img_inf = None

        nifti_path = op.join(sub_dir
            
        )

        nifti, img = load_nifti(op.join(nifti_path))

        img_tensor = niimg2tensor(img)
 
        img_tensor= torch.unsqueeze(img_tensor, 0)
        sig=torch.nn.Sigmoid()
        with torch.set_grad_enabled(False):
            x = img_tensor.to(device, dtype=torch.float32)
            x = torch.unsqueeze(x,0)
            y_inf = unet(x)

            #img_inf[i] = y_inf.detach().cpu()
            
            img_tensor_inf = y_inf.detach().cpu()
            img_tensor_inf= torch.round(sig(img_tensor_inf)) 
            
            img_inf_np = img_tensor_inf.squeeze().detach().numpy()
            img_inf_np_mask = img_inf_np[1,:,:,:]

            inf_nifti = nib.nifti1.Nifti1Image(
                img_inf_np_mask,
                nifti.affine,
                nifti.header,
            )
            #nifti_path_=op.split(nifti_path)[0] 
            outname = op.splitext(op.splitext(nifti_path)[0])[0] + "_inference_3D_Unet_RELU_FDL.nii.gz"
        
            nib.save(inf_nifti,  outname)  # TODO: sensible default



    


if __name__ == "__main__":
    main()
