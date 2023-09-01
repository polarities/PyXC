import numpy as np
import pytest

from pyxc.transform.affine2d import Affine2D


# Check if the matrix integrity checker works as expected for correct matrix input.
def test_matrix_integrity_checker_correct():
    transform = Affine2D()
    matrix = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )  # correct shape and last row is [0, 0, 1]
    assert transform._matrix_integrity_checker(matrix) is None


# Verify if the matrix integrity checker raises a ValueError for an incorrect matrix shape.
def test_matrix_integrity_checker_incorrect_shape():
    transform = Affine2D()
    matrix = np.array([[1, 2], [3, 4]])  # incorrect shape
    with pytest.raises(ValueError):
        transform._matrix_integrity_checker(matrix)


# Verify if the matrix integrity checker raises a ValueError for an incorrect matrix value.
def test_matrix_integrity_checker_incorrect_value():
    transform = Affine2D()
    matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])  # incorrect last row
    with pytest.raises(ValueError):
        transform._matrix_integrity_checker(matrix)


# Check if the 'translate' method correctly adds the translation matrix to the transformation queue.
def test_translate():
    transform = Affine2D()
    transform.translate(1, 2)
    assert transform.queue[-1] == [[1, 0, 1], [0, 1, 2], [0, 0, 1]]


# Verify if the 'rotate' method correctly adds the rotation matrix to the transformation queue.
def test_rotate():
    transform = Affine2D()
    transform.rotate(np.pi / 2)
    expected_matrix = [
        [np.cos(np.pi / 2), np.sin(np.pi / 2), 0],
        [-np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
        [0, 0, 1],
    ]
    assert np.allclose(transform.queue[-1], expected_matrix)


# Check if the 'shear_x' method correctly adds the shear matrix in the x-direction to the transformation queue.
def test_shear_x():
    transform = Affine2D()
    transform.shear_x(np.pi / 4)  # 45 degree shear in x
    expected_matrix = [[1, np.tan(np.pi / 4), 0], [0, 1, 0], [0, 0, 1]]
    assert np.allclose(transform.queue[-1], expected_matrix)


# Verify if the 'shear_y' method correctly adds the shear matrix in the y-direction to the transformation queue.
def test_shear_y():
    transform = Affine2D()
    transform.shear_y(np.pi / 4)  # 45 degree shear in y
    expected_matrix = [[1, 0, 0], [np.tan(np.pi / 4), 1, 0], [0, 0, 1]]
    assert np.allclose(transform.queue[-1], expected_matrix)


# Check if the 'scaling' method correctly adds the scaling matrix to the transformation queue.
def test_scaling():
    transform = Affine2D()
    transform.scaling(2, 3)
    expected_matrix = [[2, 0, 0], [0, 3, 0], [0, 0, 1]]
    assert np.array_equal(transform.queue[-1], expected_matrix)


# Verify if the 'flip_x' method correctly adds the flip matrix in the x-direction to the transformation queue.
def test_flip_x():
    transform = Affine2D()
    transform.flip_x()
    expected_matrix = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert np.array_equal(transform.queue[-1], expected_matrix)


# Check if the 'flip_y' method correctly adds the flip matrix in the y-direction to the transformation queue.
def test_flip_y():
    transform = Affine2D()
    transform.flip_y()
    expected_matrix = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
    assert np.array_equal(transform.queue[-1], expected_matrix)


# Verify if the 'flip_xy' method correctly adds the flip matrix in both x and y directions to the transformation queue.
def test_flip_xy():
    transform = Affine2D()
    transform.flip_xy()
    expected_matrix = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    assert np.array_equal(transform.queue[-1], expected_matrix)
