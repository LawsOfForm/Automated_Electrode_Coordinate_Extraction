import numpy as np
from scipy import ndimage
import nibabel as nib

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

def pad_tranformation_matrix(R):
    R_pad = np.eye(4)
    R_pad[:3, :3] = R
    return R_pad
    
if __name__ == "__main__":
    nifti = nib.load("/media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction/sub-010/electrode_extraction/ses-1/run-01/finalmask.nii.gz")

    centre = np.array([176, 165, 227])
    normal = np.array([89, 21, 70])

    height = 5
    radius = 10
    xdim = 224
    ydim = 288
    zdim = 288
    empty_img = np.zeros(nifti.shape)
    
    for z in range(centre[2], centre[2] + height):
        
        mask = circle_mask(xdim, ydim, radius, centre[:2])
        empty_img[:,:, z] = mask.transpose()
    
    mask_img = empty_img
    
    R = get_transformation_matrix(np.array([0, 0, 1]), normal)
    R_pad = pad_tranformation_matrix(R)
    
    new_img = nib.Nifti1Image(
        mask_img,
        nifti.affine,
        nifti.header
    )
    
    nib.save(new_img,
        "/media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction/sub-010/electrode_extraction/ses-1/run-01/cylinder_test.nii.gz")

    