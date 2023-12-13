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
    mid = []
    for elec in range(0, n_electrodes):
        _x_list = [vertex[0] for vertex in vertexes[0 + elec * 6 : 5 + elec * 6]]
        _y_list = [vertex[1] for vertex in vertexes[0 + elec * 6 : 5 + elec * 6]]
        _z_list = [vertex[2] for vertex in vertexes[0 + elec * 6 : 5 + elec * 6]]
        _len = len(vertexes[0 + elec * 6 : 5 + elec * 6])
        mid.append(
            np.array(
                [(sum(_x_list) / _len), (sum(_y_list) / _len), (sum(_z_list) / _len)]
            ).astype("int32")
        )
    return mid
