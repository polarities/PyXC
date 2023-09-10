from abc import ABCMeta, abstractmethod
from typing import Type, Optional
from warnings import warn

import numpy as np

from ..core.processor.arrays import xy2matrix


class TransformationBase:
    def __init__(self, parent: Optional["TransformationBase"] = None, *args, **kwargs):
        self.parent = parent

    def set_parent(self, parent: Type["TransformationBase"]):
        self.parent = parent

    def reset_parent(self):
        self.parent = None

    def apply_transformation(self, x, y):
        raise NotImplementedError("Transformation is not implemented.")

    def apply_reverse_transformation(self, x, y):
        raise NotImplementedError("Reverse transformation is not available.")


class MatrixTransformationBase(TransformationBase):
    """
    A base class for matrix transformations. It provides the basic functionality for matrix transformations.

    Parameters
    ----------
    transformation_matrix : np.ndarray, optional
        The transformation matrix to be used for transformation.
    parent : TransformationBase, optional
        The parent transformation object.

    Attributes
    ----------
    externally_set_transformation_matrix : np.ndarray
    transformation_matrix : np.ndarray
    reverse_transformation_matrix : np.ndarray
    parent_transformation_matrix : np.ndarray
    """

    def __init__(
        self,
        transformation_matrix: Optional[np.ndarray] = None,
        parent: Optional[TransformationBase] = None,
    ):
        super(MatrixTransformationBase, self).__init__(parent=parent)

        if transformation_matrix is not None:
            self._matrix_integrity_checker(transformation_matrix)

        # Core variables
        self._externally_set_transformation_matrix: np.ndarray | None = (
            transformation_matrix
        )
        self.queue: list = list()  # Storage of applied transformation matrices
        self.is_updated = False

        # Basic transformation matrix.
        self.identity = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )

    @property
    def externally_set_transformation_matrix(self):
        return self._identity_if_none(self._externally_set_transformation_matrix)

    @property
    def transformation_matrix(self):
        if len(self.queue) == 0:
            if self._externally_set_transformation_matrix is not None:
                return (
                    self.externally_set_transformation_matrix
                    @ self.parent_transformation_matrix
                )
            else:
                return self.identity @ self.parent_transformation_matrix
        else:
            if self._externally_set_transformation_matrix is not None:
                warn(
                    "Transformation matrix and queue were set. Queue will be prioritised."
                )
            return self._compile_from_queue() @ self.parent_transformation_matrix

    @transformation_matrix.setter
    def transformation_matrix(self, matrix):
        """
        Set the transformation matrix, checks matrix integrity before setting.

        Parameters
        ----------
        matrix : ndarray
            The matrix to set as the transformation matrix.
        """
        self._matrix_integrity_checker(matrix)

        self._externally_set_transformation_matrix = matrix
        self.is_updated = True

    @property
    def reverse_transformation_matrix(self):
        """Return a compiled reverse transformation matrix."""
        return np.linalg.pinv(self.transformation_matrix)

    @property
    def parent_transformation_matrix(self):
        """Return the base transformation matrix if set."""
        if self.parent is None:
            return self.identity
        else:
            return self._identity_if_none(self.parent.transformation_matrix)

    def _compile_from_queue(self):
        """
        Calculate the total transformation matrix based on the queue and the parent transformation matrix.

        Returns
        -------
        transform_matrix : np.ndarray
            Complex 2D image affine transformation matrix.

        Notes
        -----
        When the `_externally_set_transformation_matrix` is None, then the total transformation matrix is calculated from the
        parent and the queue. If `_externally_set_transformation_matrix` is not None, then the total transformation matrix
        is calculated from the parent and the `_externally_set_transformation_matrix` value.
        """
        self.queue.reverse()  # Transformation order should be reversed.
        transform_matrix = self.identity  # Starting with the identity matrix.
        for operation in self.queue:
            transform_matrix = transform_matrix @ operation
        self.queue.reverse()  # Reverse the queue back to its original order.
        return transform_matrix

    def reset_queue(self):
        self.queue = list()

    def apply_transformation(self, x, y):
        intermediate_matrix = self.transformation_matrix @ xy2matrix(x, y)
        return (
            intermediate_matrix[0] / intermediate_matrix[2],
            intermediate_matrix[1] / intermediate_matrix[2],
        )

    def _identity_if_none(self, value):
        """
        Return the identity matrix if the provided value is None, else returns the value itself.

        Parameters
        ----------
        value : ndarray or None
            The value to check.

        Returns
        -------
        ndarray
            Identity matrix if value is None, else the value itself.
        """

        if value is None:
            return self.identity
        else:
            return value

    @abstractmethod
    def _matrix_integrity_checker(self, matrix):
        """
        Check whether the provided matrix is valid.

        Parameters
        ----------
        matrix : ndarray
            The matrix to check.

        Returns
        -------
        bool
            Abstract method to be implemented in subclasses. Expected to return True if matrix is valid, False otherwise.
        """
        pass


class ParametricTransformationBase(TransformationBase):
    pass
