from tkinter import W
import cv2 as cv
import numpy as np


def get_rotation_matrix(a, b):
    """
    Get the rotation matrix to rotate vector a onto vector b.

    Parameters
    ----------
    a : np.ndarray
        The vector to rotate.
    b : np.ndarray
        The vector to rotate onto.

    Returns
    -------
    np.ndarray
        The rotation matrix.
    """
    u_normal = a / np.linalg.norm(a)
    v_normal = b / np.linalg.norm(b)

    v = np.cross(u_normal, v_normal)

    v_skew_symmetric = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    rotation_matrix = (
        np.eye(3)
        + v_skew_symmetric
        + np.dot(v_skew_symmetric, v_skew_symmetric)
        * 1
        / (1 + np.dot(u_normal, v_normal))
    )

    return rotation_matrix


def pad_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Pad a 3x3 matrix to a 4x4 matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The 3x3 matrix.

    Returns
    -------
    np.ndarray
        The padded 4x4 matrix.
    """
    matrix_pad = np.eye(4)
    matrix_pad[:3, :3] = matrix
    return matrix_pad


def rotate_img_obj(
    img: np.ndarray, rotation_matrix: np.ndarray, centre: np.ndarray
) -> np.ndarray:
    """
    Rotate a binary image object and return the indices of the rotated image.

    Parameters
    ----------
    img : np.ndarray
        The binary image.
    rotation_matrix : np.ndarray
        The rotation matrix.
    centre : np.ndarray
        The centre of the rotation.

    Returns
    -------
    np.ndarray
        The indeices of the rotated binary image.
    """
    centre = centre.reshape(3, 1)
    cylinder_ind = np.where(img == 1)
    centered_cylinder_ind = cylinder_ind - centre
    rotated_centered_cylinder_ind = rotation_matrix @ centered_cylinder_ind
    rotated_cylinder_ind = (rotated_centered_cylinder_ind + centre).T
    rotated_cylinder_ind = np.vstack(
        (
            np.floor(rotated_cylinder_ind),
            np.ceil(rotated_cylinder_ind),
        )
    ).astype("int32")
    return np.unique(rotated_cylinder_ind, axis=0)


def img_insert_value_at_ind(
    img: np.ndarray, inds: list, value: int | float = 1
) -> np.ndarray:
    """
    Insert a value in the img at the given indices.

    Parameters
    ----------
    img : np.ndarray
        The image in which values are inserted.
    inds : list[np.array]
        Indices at which the values are inserted.
    value : int, optional
        Value to be inserted. Default is 1.

    Returns
    -------
    np.ndarray
        Image with the inserted value
    """

    img[inds[:, 0], inds[:, 1], inds[:, 2]] = value

    return img


def fill_holes(img: np.ndarray, kernel_size: int = 3):
    """
    Fill holes in a binary image.

    Parameters
    ----------
    img : np.ndarray
        The binary image.
    kernel_size : int, optional

    Returns
    -------
    np.ndarray
        The binary image with filled holes.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    return img


def project_onto_plane(
    point: np.ndarray,
    plane_normal: np.ndarray,
    plane_point: np.ndarray,
) -> np.ndarray:
    """
    Project a point onto a plane.

    Parameters
    ----------
    point : np.ndarray
        The point to project.
    plane_normal : np.ndarray
        The normal vector of the plane.
    plane_point : np.ndarray
        A point on the plane.

    Returns
    -------
    np.ndarray
        The projected point.
    """

    v = point - plane_point

    v_parallel = (
        np.dot(v, plane_normal) / np.dot(plane_normal, plane_normal)
    ) * plane_normal

    v_ortho = v - v_parallel

    return point + v_ortho


def get_normal_component(mricoords: np.ndarray) -> np.ndarray:
    """
    Get the normal component of a plane defined by three points.

    Parameters
    ----------
    mricoords : np.ndarray
        The three points.

    Returns
    -------
    np.ndarray
        The normal component.
    """
    return np.cross(
        mricoords[1] - mricoords[0],
        mricoords[2] - mricoords[0],
    )
