#use with linux meinzer2, only loaded dataset
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms as T
import os.path as op
#from paths import root_dir
import glob
from monai.networks.nets import UNet
import torch.nn.functional as F



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
    unet = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        #channels=(32, 64, 128, 256, 512),
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        act="RELU",
    )


    path_unet = "/media/Data03/Studies/MeMoSLAP/code/Automated_Electrode_Coordinate_Extraction/electrode_extraction_from_network/Network"
    unet.load_state_dict(torch.load(op.join(path_unet, "best_metric_model_cubes.pth"), map_location='cpu'))
    unet.eval()
    unet.to(device)

    # nifti, img = load_nifti(
    #     "/media/MeMoSLAP_Subjects/SUBJECTS_XNAT/sub-4012/ses-1/anat/sub-4012_ses-1_acq-petra_run-01_PDw.nii.gz"
    # )  # TODO: path as input
    #base_dir = root_dir
    nifti_path = "/media/Data03/Thesis/Dabelstein/derivatives/cubes/"
    sub_dirs = glob.glob(f'{nifti_path}*volume*.nii.gz')

    for sub_dir in sub_dirs:

        nifti_path = op.join(sub_dir
            
        )
       
        img = None
        nifti = None
        input_image = None
        img_tensor = None
        img_tensor_inf = None
        img_inf = None

        nifti, img = load_nifti(op.join(nifti_path))
        #original_size = img.shape
        input_image = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        original_size = input_image.shape[2:]

        img_tensor= F.interpolate(input_image, size=(64,64,64), mode='trilinear')
        # Get the original image size
        
        sig=torch.nn.Sigmoid()
        with torch.set_grad_enabled(False):
            #for i, x in tqdm(enumerate(img_tensor), total=len(img_tensor)):
                #x = torch.unsqueeze(x, dim=0)
            x = img_tensor.to(device, dtype=torch.float32)
            y_inf = unet(x)

            #img_inf[i] = y_inf.detach().cpu()
            img_tensor_inf = y_inf.detach().cpu()

            img_tensor_inf= torch.round(sig(img_tensor_inf)) 

            img_inf = F.interpolate(img_tensor_inf, size=original_size, mode='trilinear')
            img_inf = torch.round(img_inf)
            # Convert the output tensor to a NumPy array
            img_inf_np = img_inf.squeeze().detach().numpy()
            img_inf_np_mask = img_inf_np[0,:,:,:]
            #img_inf_np_bg = img_inf_np[0,:,:,:]

            #np.testing.assert_almost_equal(img_inf_np_mask,1-img_inf_np_bg, decimal=5)
            #mask generieren mit 0,1 

            inf_nifti = nib.nifti1.Nifti1Image(
                img_inf_np_mask,
                nifti.affine,
                nifti.header,
            )


            outname = op.splitext(nifti_path)[0] + "_inference.nii.gz"
        
            nib.save(inf_nifti, outname)  # TODO: sensible default



    


if __name__ == "__main__":
    main()
