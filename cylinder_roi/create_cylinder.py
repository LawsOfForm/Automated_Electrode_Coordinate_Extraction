import numpy as np
import nibabel as nib
import cv2 as cv
import os.path as op
from scipy.spatial.transform import Rotation as R
import scipy

def circle_mask(xdim, ydim, radius, centre):
    Y, X = np.ogrid[:ydim, :xdim]
    dist_from_centre = np.sqrt((X - centre[0])**2 + (Y-centre[1])**2)
    
    return dist_from_centre <= radius
    
def get_theta(u, v):
    cov = np.dot(u, v)
    u_len = np.linalg.norm(u)
    v_len = np.linalg.norm(v)
    return np.arccos(cov/(u_len*v_len))

def get_transformation_matrix(a, b):
    
    u_normal = a / np.linalg.norm(a)
    v_normal = b / np.linalg.norm(b)
    
    v = np.cross(u_normal, v_normal) 
    
    v_skew_symmetric = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    
    R = (np.eye(3)
        + v_skew_symmetric
        + np.dot(v_skew_symmetric, v_skew_symmetric)
        * 1 / (1 + np.dot(u_normal, v_normal))
    )
    
    return R

def create_transformation_matrix(v1, v2):
    # Normalisieren Sie die Vektoren
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Berechnen Sie die Rotationsachse und den Winkel
    axis = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2))

    # Erstellen Sie die Rotationsmatrix
    rotation = R.from_rotvec(axis * angle)
    return rotation.as_dcm()

def pad_transformation_matrix(R):
    R_pad = np.eye(4)
    R_pad[:3, :3] = R
    return R_pad

def rotate_img_obj(img, R, centre):
    centre = centre.reshape(3, 1)
    cylinder_ind = np.where(img == 1)
    centered_cylinder_ind = cylinder_ind - centre
    rotated_centered_cylinder_ind = R @ centered_cylinder_ind
    rotated_cylinder_ind = (
        rotated_centered_cylinder_ind + centre).T
    rotated_cylinder_ind = np.vstack((
        np.floor(rotated_cylinder_ind),
        np.ceil(rotated_cylinder_ind)
    )).astype("int32")
    
    return np.unique(rotated_cylinder_ind, axis = 0)

def fill_holes(img, kernel_size = 3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    return img

def read_mricoords(path):
    return scipy.io.loadmat(path)["mricoords"].T

def get_normal_component(mricoords):
    return np.cross(
        mricoords[1] - mricoords[0],
        mricoords[2] - mricoords[0]
    )

    
if __name__ == "__main__":
    sub = "010"
    ses = "1"
    run = "01"
    root_dir = op.join(
        "/media",
        "MeMoSLAP_Subjects",
        "derivatives",
        "automated_electrode_extraction",
    )
    sub_dir = op.join(
       root_dir,
         f"sub-{sub}",
         "electrode_extraction",
         f"ses-{ses}",
         f"run-{run}",
    ) 
    nifti = nib.load(op.join(sub_dir,"finalmask.nii.gz"))
    
    mricoords = read_mricoords(op.join(sub_dir, "mricoords_1.mat")) 
    
    centre = mricoords[0]
    normal = get_normal_component(mricoords[:3]) 

    height = 5
    radius = 10
    empty_img = np.zeros(nifti.shape)
    
    for z in range(centre[2], centre[2] + height):
        
        mask = circle_mask(
            nifti.shape[0],
            nifti.shape[1], 
            radius,
            centre[:2],
        )
        empty_img[:,:, z] = mask.transpose()
    
    mask_img = empty_img
    
    rotation_matrix = get_transformation_matrix(np.array([0, 0, 1]), normal)
    
    rotated_cylinder_ind = rotate_img_obj(
        mask_img,
        rotation_matrix, 
        centre
    )
    
    emtpy_img = np.zeros((nifti.shape))
    
    emtpy_img[
        rotated_cylinder_ind[:, 0],
        rotated_cylinder_ind[:, 1],
        rotated_cylinder_ind[:, 2]
    ] = 1
    
    rotated_cylinder = fill_holes(emtpy_img) 
     
    new_img = nib.Nifti1Image(
        rotated_cylinder,
        nifti.affine,
        nifti.header
    )
    
    nib.save(new_img,
        op.join(sub_dir, "cylinder_test.nii.gz"))
