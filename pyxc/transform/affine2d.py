import numpy as np

from .transform_base import MatrixTransformationBase
from typing import Type, Optional


class Affine2D(MatrixTransformationBase):
    """
    2-dimensional Affine transformation.

    By calling each transformation method, such as translate(), corresponding transformation matrix is generated and
    queued in the transformation queue. The queued transformation entries are later compiled into a single matrix by
    calling the compile() method.

    Parameters
    ----------
    transformation_matrix : np.ndarray, optional
        The transformation matrix to be used for the transformation.
    parent : TransformationBase, optional
        The parent transformation object.

    Methods
    -------
    translate(tx, ty)
    rotate(angle)
    shear_x(angle)
    shear_y(angle)
    scale(x_scale, y_scale)
    flip_x()
    flip_y()
    flip_xy()
    """

    def __init__(
        self,
        transformation_matrix: Optional[np.ndarray] = None,
        parent: Optional[MatrixTransformationBase] = None,
    ):
        super(Affine2D, self).__init__(transformation_matrix, parent)

    def _matrix_integrity_checker(self, matrix):
        """
        Check whether the shape of the provided matrix is right.

        Parameters
        ----------
        matrix : np.ndarray
            The matrix to check.
        """
        if np.shape(matrix) != (3, 3):
            raise ValueError("Shape mismatch.")

        if not np.array_equal(matrix[2], np.array([0, 0, 1])):
            raise ValueError(
                "Error: The third row of the affine transformation matrix should be (0, 0, 1)."
            )

    def translate(self, tx: float, ty: float):
        """
        Transformation matrix.

        Parameters
        ----------
        tx : float
            The amount of the translation along x direction.
        ty : float
            The amount of the translation along y direction.

        Returns
        -------
        np.ndarray:
            Translation matrix.

        See Also
        --------
        pyxc.transform.transform_base.TransformationCommon.queue_return_handler: Possible keyword arguments.
        """
        transformation_matrix = [[1, 0, tx], [0, 1, ty], [0, 0, 1]]

        self.queue.append(transformation_matrix)

    def rotate(self, angle: float):
        """
        Rotation matrix.

        Parameters
        ----------
        angle : float
            Rotation angle in radian. Note: numpy.deg2rad() is the convenient method to convert degree to radian.

        Returns
        -------
        np.ndarray:
            Rotation matrix.

        See Also
        --------
        pyxc.transform.transform_base.TransformationCommon.queue_return_handler: Possible keyword arguments.
        """
        transformation_matrix = [
            [np.cos(angle), np.sin(angle), 0],
            [-1 * np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
        self.queue.append(transformation_matrix)

    def shear_x(self, angle: float):
        """
        Shear transformation along the x-axis.

        Parameters
        ----------
        angle : float
            Shear angle in radian. Note: numpy.deg2rad() is the convenient method to convert degree to radian.

        Returns
        -------
        np.ndarray:
            Shear matrix along the x-direction.

        See Also
        --------
        pyxc.transform.transform_base.TransformationCommon.queue_return_handler: Possible keyword arguments.
        """
        transformation_matrix = [[1, np.tan(angle), 0], [0, 1, 0], [0, 0, 1]]
        self.queue.append(transformation_matrix)

    def shear_y(self, angle: float):
        """
        Shear transformation along the y-axis.

        Parameters
        ----------
        angle : float
            Shear angle in radian. Note: numpy.deg2rad() is the convenient method to convert degree to radian.

        Returns
        -------
        np.ndarray:
            Shear matrix along the x-direction.

        See Also
        --------
        pyxc.transform.transform_base.TransformationCommon.queue_return_handler: Possible keyword arguments.
        """
        transformation_matrix = [[1, 0, 0], [np.tan(angle), 1, 0], [0, 0, 1]]
        self.queue.append(transformation_matrix)

    def scaling(self, sx: float, sy: float):
        """
        Scaling transformation matrix.

        Parameters
        ----------
        sx : float
            Scaling factor along the x-direction.
        sy : float
            Scaling factor along the y-direction.

        Returns
        -------
        np.ndarray:
            Scaling matrix along the x-direction.

        See Also
        --------
        pyxc.transform.transform_base.TransformationCommon.queue_return_handler: Possible keyword arguments.

        Notes
        -----
        If `sx` = `sy` = 1, then the composed matrix is equal to the identity matrix.
        """
        transformation_matrix = [[sx, 0, 0], [0, sy, 0], [0, 0, 1]]
        self.queue.append(transformation_matrix)

    def flip_x(self):
        """
        Invert by x-axis.

        Returns
        -------
        np.ndarray:
            Reflection matrix along the x-direction.
        """
        transformation_matrix = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.queue.append(transformation_matrix)

    def flip_y(self):
        """
        Invert by y-axis.

        Returns
        -------
        np.ndarray:
            Reflection matrix along the y-direction.
        """
        transformation_matrix = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        self.queue.append(transformation_matrix)

    def flip_xy(self):
        """
        Invert X- and Y-axis signs.

        Returns
        -------
        np.ndarray:
            Reflection matrix along the x- and y-direction.

        See Also
        --------
        pyxc.transform.transform_base.TransformationCommon.queue_return_handler: Possible keyword arguments.
        flip_x, flip_y
        """
        transformation_matrix = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        self.queue.append(transformation_matrix)
