import numpy as np
import pytest

from pyxc.transform.homography import Homography


# Check if the matrix integrity checker works as expected for a correct matrix input with the right shape and value.
def test_matrix_integrity_checker_correct_shape_and_value():
    homography = Homography()
    matrix = np.array([[1, 0, 0], [1, 1, 0], [0, 0.5, 1]])  # shape (3, 3) and M_33 = 1
    assert (
        homography._matrix_integrity_checker(matrix) is None
    )  # Expect no exception to be raised


# Verify if the matrix integrity checker raises a ValueError for an incorrect matrix shape.
def test_matrix_integrity_checker_incorrect_shape():
    homography = Homography()
    matrix = np.array([[1, 0], [0, 1]])  # shape (2, 2) instead of (3, 3)
    with pytest.raises(ValueError) as context:
        homography._matrix_integrity_checker(matrix)


# Verify if the matrix integrity checker raises a ValueError for an incorrect matrix value.
def test_matrix_integrity_checker_incorrect_value():
    homography = Homography()
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # shape (3, 3) but M_33 != 1
    with pytest.raises(ValueError) as context:
        homography._matrix_integrity_checker(matrix)
