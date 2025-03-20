import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from pathlib import Path
import pandas as pd
import os

def create_sphere(shape, center, radius):
    """Create a binary sphere mask."""
    grid = np.indices(shape)
    dist = np.sqrt(np.sum((grid - np.array(center)[:, None, None, None])**2, axis=0))
    sphere = (dist <= radius).astype(float)
    return sphere

def overlay_red_sphere_on_mri(mri_data, sphere, alpha=0.5):
    """Overlay a red transparent sphere on the MRI data."""
    # Convert grayscale MRI to RGB
    mri_rgb = np.stack([mri_data] * 3, axis=-1)

    # Create a red sphere mask
    red_sphere = np.zeros(mri_data.shape + (3,))
    red_sphere[sphere > 0] = [1, 0, 0]  # Red color

    # Overlay the red sphere with transparency
    overlayed_data = mri_rgb.copy()
    overlayed_data[sphere > 0] = overlayed_data[sphere > 0] * (1 - alpha) + red_sphere[sphere > 0] * alpha

    return overlayed_data

def mni_to_voxel(mni_coords, affine):
    """Convert MNI coordinates (real-world) to voxel coordinates."""
    # Add a 1 to the MNI coordinates for affine transformation
    mni_coords = np.append(mni_coords, 1)
    # Compute the voxel coordinates using the inverse affine matrix
    voxel_coords = np.linalg.inv(affine) @ mni_coords
    # Return the first three elements (x, y, z) as integers
    return np.round(voxel_coords[:3]).astype(int)

def save_3_slices(mri_rgb_1, voxel_coords_1, mni_coords_1, mri_rgb_2, voxel_coords_2, mni_coords_2, Method_str_1, Method_str_2, Subject,Session, Run, Electrode, output_path):
    """Save axial, sagittal, and coronal slices as one PNG."""
    # Extract slices
    axial_slice_1 = mri_rgb_1[:, :, voxel_coords_1[2], :]  # Axial (z-axis)
    sagittal_slice_1 = mri_rgb_1[:, voxel_coords_1[1], :, :]  # Sagittal (y-axis)
    coronal_slice_1 = mri_rgb_1[voxel_coords_1[0], :, :, :]  # Coronal (x-axis)
    axial_slice_2 = mri_rgb_2[:, :, voxel_coords_2[2], :]  # Axial (z-axis)
    sagittal_slice_2 = mri_rgb_2[:, voxel_coords_2[1], :, :]  # Sagittal (y-axis)
    coronal_slice_2 = mri_rgb_2[voxel_coords_2[0], :, :, :]  # Coronal (x-axis)

    # Rotate sagittal and coronal slices 90 degrees to the left
    axial_slice_1 = np.rot90(axial_slice_1, k=1)  # Rotate 90 degrees counterclockwise
    sagittal_slice_1 = np.rot90(sagittal_slice_1, k=1)  # Rotate 90 degrees counterclockwise
    coronal_slice_1 = np.rot90(coronal_slice_1, k=1)  # Rotate 90 degrees counterclockwise
    axial_slice_2 = np.rot90(axial_slice_2, k=1)  # Rotate 90 degrees counterclockwise
    sagittal_slice_2 = np.rot90(sagittal_slice_2, k=1)  # Rotate 90 degrees counterclockwise
    coronal_slice_2 = np.rot90(coronal_slice_2, k=1)  # Rotate 90 degrees counterclockwise


    # Create a figure with 3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(7, 5))

    # Display axial slice
    axes[0,0].imshow(axial_slice_1)
    axes[0,0].set_title(f"Axial Slice (z={mni_coords_1[2].round(2)})")
    axes[0,0].axis('off')

    # Display sagittal slice
    axes[0,1].imshow(sagittal_slice_1)
    axes[0,1].set_title(f"({Subject}_{Session}_{Run}_{Electrode})\n {Method_str_1} \n Sagittal Slice (y={mni_coords_1[1].round(2)})")
    axes[0,1].axis('off')

    # Display coronal slice
    axes[0,2].imshow(coronal_slice_1)
    axes[0,2].set_title(f"Coronal Slice (x={mni_coords_1[0].round(2)})")
    axes[0,2].axis('off')

        # Display axial slice
    axes[1,0].imshow(axial_slice_2)
    axes[1,0].set_title(f"Axial Slice (z={mni_coords_2[2].round(2)})")
    axes[1,0].axis('off')

    # Display sagittal slice
    axes[1,1].imshow(sagittal_slice_2)
    axes[1,1].set_title(f"{Method_str_2} \n Sagittal Slice (y={mni_coords_2[1].round(2)})")
    axes[1,1].axis('off')

    # Display coronal slice
    axes[1,2].imshow(coronal_slice_2)
    axes[1,2].set_title(f"Coronal Slice (x={mni_coords_2[0].round(2)})")
    axes[1,2].axis('off')

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def main(mri_path, Coordinates_1, Coordinates_2, Method_str_1, Method_str_2,Subject,Session, Run, Electrode,output_path):
    # Load the MRI image
    mri_path = mri_path
    mri_img = nib.load(mri_path)
    mri_data = mri_img.get_fdata()
    affine = mri_img.affine  # Get the affine transformation matrix

    # Normalize MRI data to [0, 1] for visualization
    mri_data = (mri_data - mri_data.min()) / (mri_data.max() - mri_data.min())

    # Define the MNI coordinates and sphere radius
    #mni_coords = [76, -51, -13]  # Replace with your MNI coordinates fully automated
    mni_coords_1 = [Coordinates_1[0],Coordinates_1[1], Coordinates_1[2]]  # Replace with your MNI coordinates
    mni_coords_2 = [Coordinates_2[0],Coordinates_2[1], Coordinates_2[2]]  # Replace with your MNI coordinates
      
  
    radius = 2  # Replace with your desired radius

    # Convert MNI coordinates to voxel coordinates
    voxel_coords_1 = mni_to_voxel(mni_coords_1, affine)
    voxel_coords_2 = mni_to_voxel(mni_coords_2, affine)
    print("MNI coordinates_1:", mni_coords_1)
    print("Voxel coordinates_1:", voxel_coords_1)
    print("MNI coordinates_2:", mni_coords_2)
    print("Voxel coordinates_2:", voxel_coords_2)

    # Check if the voxel coordinates are within the MRI volume
    if not all(0 <= coord < dim for coord, dim in zip(voxel_coords_1, mri_data.shape)):
        raise ValueError("Voxel coordinates are outside the MRI volume!")
    
    # Check if the voxel coordinates are within the MRI volume
    if not all(0 <= coord < dim for coord, dim in zip(voxel_coords_2, mri_data.shape)):
        raise ValueError("Voxel coordinates are outside the MRI volume!")

    # Create the sphere
    sphere_1 = create_sphere(mri_data.shape, voxel_coords_1, radius)
    sphere_2 = create_sphere(mri_data.shape, voxel_coords_2, radius)

    # Print the number of voxels in the sphere for debugging
    print("Number of voxels in the sphere:", np.sum(sphere_1))
    print("Number of voxels in the sphere:", np.sum(sphere_2))

    # Overlay the red sphere on the MRI data
    overlayed_data_1 = overlay_red_sphere_on_mri(mri_data, sphere_1, alpha=0.5)
    overlayed_data_2 = overlay_red_sphere_on_mri(mri_data, sphere_2, alpha=0.5)

    # Save the 3 slices (axial, sagittal, coronal) as one PNG
    output_path = output_path
    save_3_slices(overlayed_data_1, voxel_coords_1, mni_coords_1,overlayed_data_2, voxel_coords_2, mni_coords_2, Method_str_1, Method_str_2, Subject,Session, Run, Electrode,output_path)

    print(f"3 slices saved to {output_path}")

def read_outlier_LoA(Outlier_path, Method1, Method2, Coord):
    file_name=f'points_outside_LoA_{Method1}_{Method2}_{Coord}.csv'
    df = pd.read_csv(os.path.join(Outlier_path,file_name))
    list=df[['Subject','Session','run','Electrode']].values
    return list

def get_coordinates(root_path, Subject, Session, Run,Electrode,Method):
    if Method == 'semi-automated':
        Method = 'half-automated'

    if Method != 'full-automated':
        Rater = 'Kira'
    else:
        Rater = 'Network'

    #read in corrected coordinates
    df = pd.read_csv(os.path.join(root_path,'tables','corrected_electrode_positions.csv'))
    X = df[(df['Subject']==Subject) & 
           (df['Session']== Session) & 
           (df['run']== Run) & 
           (df['Electrode']== Electrode) & 
           (df['Rater']==Rater) & 
           (df['Method']==Method) & 
           (df['Dimension']=='X')]['Coordinates'].values[0]
    Y = df[(df['Subject']==Subject) & 
           (df['Session']== Session) & 
           (df['run']== Run) & 
           (df['Electrode']== Electrode) & 
           (df['Rater']==Rater) & 
           (df['Method']==Method) & 
           (df['Dimension']=='Y')]['Coordinates'].values[0]
    Z = df[(df['Subject']==Subject) & 
           (df['Session']== Session) & 
           (df['run']== Run) & 
           (df['Electrode']== Electrode) & 
           (df['Rater']==Rater) & 
           (df['Method']==Method) & 
           (df['Dimension']=='Z')]['Coordinates'].values[0]
      
    Coordinates = [X,Y,Z]
    return Coordinates

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = Path(__file__).resolve().parent

    # Convert the path to a string (if needed) and split it into parts
    path_parts = script_dir.parts

    # Remove the last folder from the path
    root_path= Path(*path_parts[:-1])

    # get List of subject, session, run from csv list
    Outlier_path = os.path.join(root_path,"tables")
    Method = ['full-automated','semi-automated', 'manually']
    
    #Choose pair of methods
    Method_str_1 = Method[1]
    Method_str_2 = Method[2]

    Coord = ['Euclidean_Norm', 'X','Y','Z']
    # Choose
    list = read_outlier_LoA(Outlier_path,  Method_str_1,  Method_str_2, Coord[0])

    for elements in list:
        Subject = elements[0]
        Session = elements[1]
        Run = elements[2]
        Electrode = elements[3]
    
        mri_path = f'/media/Data03/Thesis/Hering/derivatives/automated_electrode_extraction/{Subject}/electrode_extraction/{Session}/{Run}/petra_.nii.gz'
        
        Coordinates_1 = get_coordinates(root_path, Subject,Session, Run, Electrode, Method_str_1)
        Coordinates_2 = get_coordinates(root_path, Subject,Session, Run, Electrode, Method_str_2)

        output_path = f'/media/Data03/Projects/Paper_Method_Auto_Semi_Manual/figures/Outlier_Loa/Outlier_{Subject}_{Session}_{Run}_{Electrode}_{Method_str_1}_{Method_str_2}.png'
        main(mri_path, Coordinates_1, Coordinates_2, Method_str_1, Method_str_2, Subject,Session, Run, Electrode, output_path)