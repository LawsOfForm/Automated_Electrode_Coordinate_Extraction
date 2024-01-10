import numpy as np


def circle_mask(xdim, ydim, radius, centre):
    """
    Create a circle mask.

    Parameters
    ----------
    xdim : int
        The x dimension of the mask.
    ydim : int
        The y dimension of the mask.
    radius : int
        The radius of the circle.
    centre : np.ndarray
        The centre of the circle.

    Returns
    -------
    np.ndarray
        The circle mask.
    """
    Y, X = np.ogrid[:ydim, :xdim]
    dist_from_centre = np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2)

    return dist_from_centre <= radius


def cylinder(img, centre, radius, height):
    """
    Create a cylinder mask.

    Parameters
    ----------
    img : np.ndarray
        The image from which the dimension of the mask is infered.
    centre : np.ndarray
        The centre of the cylinder.
    radius : int
        The radius of the cylinder.
    height : int
        The height of the cylinder.

    Returns
    -------
    np.ndarray
        The cylinder mask.
    """
    cylinder_img = np.zeros(img.shape)

    start = centre[2]
    stop = centre[2] + height

    range_stride = 1 if start < stop else -1

    for z in range(start, stop, range_stride):
        mask = circle_mask(
            img.shape[0],
            img.shape[1],
            radius,
            centre[:2],
        )
        cylinder_img[:, :, z] = mask.transpose()

    return cylinder_img


def centroid(vertexes, n_electrodes):
    """
    Calculate the centroid of a set of vertexes. The vertexes are assumed to be
    ordered in the following way: vertexes for electrode 1, vertexes for
    electrode 2, ..., vertexes for electrode n.

    Parameters
    ----------
    vertexes : np.ndarray
        The vertexes. Shape: (n_electrodes * number of verteces per electrodes, 3)
        last dimension: x, y, z.
    n_electrodes : int
        The number of electrodes.

    Returns
    -------
    np.ndarray
        The centroid.
    """
    vertexes = vertexes.reshape(n_electrodes, -1, 3)
    return np.mean(vertexes, axis=1).astype("int32")
