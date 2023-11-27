import numpy as np
import scipy


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
