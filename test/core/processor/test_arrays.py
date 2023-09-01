import numpy as np
import pytest

from pyxc.core.processor.arrays import (
    reshape_if_1dim,
    image_notator,
    image_serializer,
    get_structured_array_field_names_from_indices,
    xy2matrix,
    matrix2xy,
)


# Test function for the to check its behavior for different input arrays.
def test_reshape_if_1dim():
    # test for 1-dimensional array
    array_1d = np.array([1, 2, 3, 4])
    reshaped_array = reshape_if_1dim(array_1d)
    assert reshaped_array.shape == (4, 1)
    # test for 2-dimensional array
    array_2d = np.array([[1, 2], [3, 4]])
    same_array = reshape_if_1dim(array_2d)
    assert np.array_equal(same_array, array_2d)
    # test for high-dimensional array
    array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    with pytest.raises(ValueError):
        reshape_if_1dim(array_3d)


# Pytest fixture that provides random 2D and 3D image arrays for testing purposes.
@pytest.fixture(params=["2d", "3d"])
def image_provider(dimension):
    if dimension == "2d":
        return np.random.randint(0, 256, (64, 64))

    elif dimension == "3d":
        return np.random.randint(0, 256, (64, 64, 64))


# Test function to verify if it correctly annotates the coordinates of a 2D image.
def test_image_notator():
    # test for 2D image
    image_2d = np.array([[1, 2], [3, 4]])
    x, y, im = image_notator(image_2d)
    assert np.array_equal(im, image_2d)
    assert np.array_equal(x, np.array([[0, 1], [0, 1]]))
    assert np.array_equal(y, np.array([[0, 0], [1, 1]]))


# Test function to verify if it correctly flattens 2D and 3D image arrays.
def test_image_serializer():
    # test for 2D image
    image_2d = np.array([[1, 2], [3, 4]])
    flattened = image_serializer(image_2d)
    assert np.array_equal(flattened, np.array([1, 2, 3, 4]))

    # test for 3D image
    image_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    flattened = image_serializer(image_3d)
    assert np.array_equal(flattened, np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))


# Test function to verify if it correctly retrieves field names from a structured NumPy array based on given field indices.
def test_get_structured_array_field_names_from_indices():
    dtype_object = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
    names = get_structured_array_field_names_from_indices(dtype_object, [0, 2])
    assert names == ("x", "z")


# Test function to verify if it correctly transforms XY coordinate pairs to a 2D matrix.
def test_xy2matrix():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    matrix = xy2matrix(x, y)
    assert np.array_equal(matrix, np.array([[1, 2, 3], [4, 5, 6], [1, 1, 1]]))


# Test function to verify if it correctly transforms a 2D matrix to XY coordinate pairs.
def test_matrix2xy():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x, y = matrix2xy(matrix)
    assert np.array_equal(x, np.array([1, 2, 3]))
    assert np.array_equal(y, np.array([4, 5, 6]))
