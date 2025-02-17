import os
import glob
import argparse
import torch
import monai.transforms as tfms
from monai.networks.nets import AttentionUnet
import nibabel as nib
import numpy as np
from pathlib import Path

# define most import path variables
script_directory = Path(__file__).parent.resolve()
root = script_directory.parent.parent.resolve()

# change the path to your image folder in which all images are stored in sub-directories
#root_images = '/media/Data03/Thesis/Hering/derivatives/automated_electrode_extraction'
root_images = '/media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction'

# choose a model which you want to use for inference
#model = "best_metric_model_0756_0402_5level_Adamax.pth" #94% of electrdoe detection rate
model = "best_metric_model_0663_1202_5level_Adam_52tr.pth"
model_path = os.path.join(root,'code','Network','models', model )
#model_path = os.path.join(root,'code','Network','models', "best_metric_model_0756_0402_5level_Adamax.pth")


def load_image(image_path):
    """Load and preprocess the input image, preserving the affine matrix."""
    # Load the image using nibabel to get the affine matrix
    nii_img = nib.load(image_path)
    affine = nii_img.affine  # Store the affine matrix
    header = nii_img.header  # Store the header (optional, for additional metadata)

    # Use MONAI to load and preprocess the image
    transform = tfms.Compose([
        tfms.LoadImage(image_only=True),
        tfms.ScaleIntensity(),
        tfms.EnsureChannelFirst(),
    ])
    image = transform(image_path)

    return image, affine, header

def get_model(model_path):
    """Load the trained model."""
    # Automatically select device: GPU if available, otherwise CPU
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If the code throughs an error uncomment the line below and comment the line above.
    device = torch.device("cpu")  # Use CPU for inference
    # if you change the network or some parameters, you need to modify the code here accordingly.
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        kernel_size=3,
        up_kernel_size=3,
        dropout=0.2,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    return model

def segment_image(model, image):
    """Perform segmentation on the input image and return a 3D volume."""
    device = next(model.parameters()).device
    with torch.no_grad():
        input_tensor = torch.unsqueeze(torch.as_tensor(image).to(device), 0)  # Add batch dimension
        output = model(input_tensor)
        print(f"Model output shape: {output.shape}")  # Should be (1, channels, depth, height, width)
    
    post_pred = tfms.Compose([tfms.AsDiscrete(argmax=True)])
    segmentation = post_pred(output[0])  # Shape: (depth, height, width)
    print(f"Segmentation shape after argmax: {segmentation.shape}")  # Should be (depth, height, width)

    # Remove the batch dimension to get a 3D tensor
    segmentation = segmentation.squeeze(0)
    print(f"3D Segmentation shape: {segmentation.shape}")

    return segmentation.cpu().numpy()

def save_segmentation(segmentation, output_path, affine):
    """Save the segmentation as a NIfTI file, preserving the affine matrix."""
    if segmentation.ndim != 3:
        raise ValueError(f"Segmentation must be 3D, but got shape: {segmentation.shape}")
    
    # Save the segmentation with the original affine matrix
    nib.save(nib.Nifti1Image(segmentation.astype(np.uint8), affine), output_path)

def main(root_images, model_path, overwrite = False):
    # Paths
    # Load the model
    model = get_model(model_path)

    # Loop through each subject directory
    for subject_dir in glob.glob(os.path.join(root_images, 'sub-*')):
        #subject = os.path.basename(subject_dir)
        #unzipped_dir = os.path.join(subject_dir, 'electrode_extraction','ses*','run*')
        unzipped_dir = os.path.join(subject_dir, 'unzipped')

        # Find all matching NIfTI files in the unzipped directory
        #nifti_files = glob.glob(os.path.join(unzipped_dir, 'petra_.nii.gz'))
        nifti_files = glob.glob(os.path.join(unzipped_dir, 'rsub*_PDw.nii'))

        if nifti_files:
            for nifti_file in nifti_files:
                # Construct the output path for the segmentation
                #output_segmentation_path = nifti_file.replace('_.nii', '_inference.nii')
                output_segmentation_path = nifti_file.replace('.nii', '_inference.nii.gz')

                # Check if the output file already exists
                if os.path.exists(output_segmentation_path) and not args.overwrite:
                    print(f"Skipping {nifti_file} because {output_segmentation_path} already exists.")
                    continue

                # Load and preprocess the image, preserving the affine matrix
                image, affine, _ = load_image(nifti_file)

                # Perform segmentation
                segmentation = segment_image(model, image)
                print(f'Shape of segmentation for {nifti_file}: {segmentation.shape}')

                # Save the segmentation result with the original affine matrix
                save_segmentation(segmentation, output_segmentation_path, affine)
                print(f"Segmentation saved to {output_segmentation_path}")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Run inference on NIfTI images.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing inference files if they exist.",
    )
    args = parser.parse_args()

    # Run the main function with the parsed arguments
    # python script.py --overwrite
    main(root_images, model_path, args.overwrite)