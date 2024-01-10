import logging
from itertools import product

import numpy as np
from util.io import load_nifti, save_nifti


def project_mask(mask, axis):
    """
    Project a mask onto an axis.

    Parameters
    ----------
    mask : np.ndarray
        The mask to project
    axis : int
        The axis to project onto

    Returns
    -------
    np.ndarray
        The projected mask
    """
    axis = [i for i in range(len(mask.shape)) if i != axis]
    proj = np.max(mask, axis=tuple(axis))
    return proj


def fast_divide_mask(mask):
    """
    Divide a mask into smaller masks by finding no overlap in the x, y, or z
    direction.

    Recursively call the function on the smaller masks until no more divisions are
    found.

    This will miss some elements, but is much faster than the slow_divide_mask
    function.

    Parameters
    ----------
    mask : np.ndarray
        The mask to divide

    Returns
    -------
    list[np.ndarray]
        A list of masks
    """
    x_proj = project_mask(mask, 0)
    y_proj = project_mask(mask, 1)
    z_proj = project_mask(mask, 2)

    x_nonzero = np.nonzero(x_proj)
    y_nonzero = np.nonzero(y_proj)
    z_nonzero = np.nonzero(z_proj)

    x_diff = np.diff(x_nonzero)
    y_diff = np.diff(y_nonzero)
    z_diff = np.diff(z_nonzero)

    idx = np.where(x_diff > 1)[1] + 1
    idy = np.where(y_diff > 1)[1] + 1
    idz = np.where(z_diff > 1)[1] + 1

    idx = np.array([x_nonzero[0][i] for i in idx], dtype=np.int32)
    idy = np.array([y_nonzero[0][i] for i in idy], dtype=np.int32)
    idz = np.array([z_nonzero[0][i] for i in idz], dtype=np.int32)

    if not any((i.size > 0 for i in [idx, idy, idz])):
        return mask

    m_shape = mask.shape

    def insert_start_end(arr, dim_shape):
        arr = np.insert(arr, 0, 0)
        arr = np.insert(arr, len(arr), dim_shape)
        return arr

    idx = insert_start_end(idx, m_shape[0])
    idy = insert_start_end(idy, m_shape[1])
    idz = insert_start_end(idz, m_shape[2])

    imgs = []

    for previous, current in zip(idx, idx[1:]):
        img = np.zeros(m_shape)
        img[previous:current, :, :] = mask[previous:current, :, :]

        if img.sum() == 0:
            continue

        for prev_y, curr_y in zip(idy, idy[1:]):
            img_y = np.zeros(m_shape)
            img_y[:, prev_y:curr_y, :] = img[:, prev_y:curr_y, :]

            if img_y.sum() == 0:
                continue

            for prev_z, curr_z in zip(idz, idz[1:]):
                img_z = np.zeros(m_shape)
                img_z[:, :, prev_z:curr_z] = img_y[:, :, prev_z:curr_z]

                if img_z.sum() == 0:
                    continue

                imgs.append(fast_divide_mask(img_z))

    return imgs


def slow_divide_mask(mask_to_div, max_element_size: int):
    """
    Divide a mask into smaller masks by finding the largest distance between
    two points in the mask and dividing the mask into two parts.

    Parameters
    ----------
    mask : np.ndarray
        The mask to divide
    max_element_size : int
        The maximum size of an element in the mask

    Returns
    -------
    list[np.ndarray]
        A list of masks
    """
    mask_idx = np.vstack(np.where(mask_to_div == 1)).T

    dist = np.zeros(len(mask_idx) ** 2)
    print(
        "Calculating distance between all points...\nThis will take some time."
    )
    for idx, (coord1, coord2) in enumerate(product(mask_idx, mask_idx)):
        dist[idx] = np.linalg.norm(coord1 - coord2)

    distr = dist.reshape(len(mask_idx), len(mask_idx))

    def get_elms(distr, elms=None, indices=None, thresversion="max"):
        """
        Get the elements in the mask that are separated by a distance larger
        than the threshold.

        The threshold is the largest distance between two points in the mask
        divided by two.

        If elements found are larger than the maximum element size, the
        function is called recursively.


        Parameters
        ----------
        distr : np.ndarray
            The distance matrix
        elms : list[np.ndarray]
            The list of elements
        indices : np.ndarray
            The indices to consider

        Returns
        -------
        list[np.ndarray]
            The list of elements
        """

        if elms is None:
            elms = []

        if indices is None:
            indices = np.arange(distr.shape[0])

        distr_finder = distr.copy()

        no_idx_found = True
        start_idx = 0

        while no_idx_found:
            max_distr_x, max_distr_y = np.where(
                distr_finder == np.max(distr_finder)
            )
            max_distr_x = max_distr_x[0]
            max_distr_y = max_distr_y[0]
            if max_distr_x in indices:
                start_idx = max_distr_x
                no_idx_found = False
                break
            distr_finder[max_distr_x, max_distr_y] = 0

        idx = start_idx

        if thresversion == "max":
            thres = np.max(distr[idx]) / 2
        elif thresversion == "hist":
            thres, _ = np.histogram(distr[idx])
            thres = np.cumsum(thres)
            thres = np.bincount(thres)
            thres = np.max(thres) - 1

        elm_idx = np.where(distr[0] < thres)[0]

        elms.append(elm_idx)

        elm = np.where(distr[0] >= thres)[0]

        if len(elm) < max_element_size:
            elms.append(elm)
            return elms

        indices_new = np.array([i for i in indices if i not in elm_idx])

        if len(indices) == len(indices_new):
            logging.warning(
                "Could not divide mask further, even though it is larger "
                "than the maximum element size."
            )
            elms.append(elm)
            return elms

        indices = indices_new

        return get_elms(distr, elms, indices)

    elms = get_elms(distr)

    sep_masks = []

    for elm in elms:
        nonzero = mask_idx[elm]
        sep_mask = np.zeros(mask_to_div.shape)
        for coord in nonzero:
            sep_mask[tuple(coord)] = 1
        sep_masks.append(sep_mask)

    return sep_masks


def recursive_flatten(l, outlist=None):
    """
    Flatten a list recursively.

    Parameters
    ----------
    l : list
        The list to flatten
    outlist : list
        The output list

    Returns
    -------
    list
        The flattened list
    """

    if outlist is None:
        outlist = []

    for item in l:
        if hasattr(item, "shape"):
            outlist.append(item)
        else:
            recursive_flatten(item, outlist)

    return outlist


if __name__ == "__main__":
    MAX_ELEMENT_SIZE = 4_000

    mask_nifti, mask = load_nifti("cylinder_plus_plug_ROI.nii.gz")

    masks = fast_divide_mask(mask)
    masks = recursive_flatten(masks)

    for i, m in enumerate(masks):
        if np.sum(m) < MAX_ELEMENT_SIZE:
            save_nifti(m, f"mask_{i}.nii.gz", mask_nifti)
            continue

        separate_masks = slow_divide_mask(m, MAX_ELEMENT_SIZE)
        for j, separate_mask in enumerate(separate_masks):
            save_nifti(separate_mask, f"mask_{i}_{j}.nii.gz", mask_nifti)
