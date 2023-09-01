import numpy as np
from numpy.typing import ArrayLike
from .transform_base import MatrixTransformationBase


class Homography(MatrixTransformationBase):
    def __init__(self, *args, **kwargs):
        super(Homography, self).__init__(*args, **kwargs)

    def _matrix_integrity_checker(self, matrix: ArrayLike):
        """
        Check whether the shape of the provided matrix is right.

        Parameters
        ----------
        matrix : np.ndarray
            The matrix to check.
        """
        if np.shape(matrix) != (3, 3):
            raise ValueError("Error: Shape mismatch.")

        if matrix[2, 2] != 1:
            raise ValueError(
                "Error: Homography transformation matrix M_33 should be 1."
            )
