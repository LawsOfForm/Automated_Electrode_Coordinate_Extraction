import nibabel as nib
import numpy as np
import scipy


def load_nifti(
    path: str,
) -> tuple[nib.filebasedimages.FileBasedImage, np.ndarray]:
    """
    Load a nifti file and return the image and the nifti object

    Parameters
    ----------
    path : str
        Path to the nifti file

    Returns
    -------
    tuple[nib.filebasedimages.FileBasedImage, np.ndarray]
        Tuple containing the nifti object and the image
    """
    nifti = nib.load(path)
    img = nifti.dataobj
    return (nifti, np.array(img))


def save_nifti(
    img: np.ndarray, path: str, ref: nib.filebasedimages.FileBasedImage
) -> None:
    """
    Save a nifti file

    Parameters
    ----------
    img : np.ndarray
        Image to save
    path : str
        Path to save the image
    ref : nib.filebasedimages.FileBasedImage
        Reference nifti object to copy the header and affine from

    Returns
    -------
    None
    """
    new_img = nib.nifti1.Nifti1Image(
        img,
        ref.affine,
        ref.header,
    )
    nib.save(new_img, path)


def read_mricoords(path: str) -> np.ndarray:
    """
    Read the mricoords from a matlab file.

    Parameters
    ----------
    path : str
        The path to the matlab file.

    Returns
    -------
    np.ndarray
        The mricoords.
    """
    return scipy.io.loadmat(path)["mricoords"].T
