from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from pyxc.core.container import Container2D
from ..core.processor.arrays import (
    image_notator,
    image_serializer,
    reshape_if_1dim,
    xyd_splitter,
)
from numpy.typing import ArrayLike


class DataLoaderBase(ABC):
    """
    An abstract base class for data loaders.

    Initializes a container with preprocessed data and returns the container when called. Works with a Layer object.
    Subclasses should implement the `prep` method to specify the preprocessing of data.

    Parameters
    ----------
    container : Container2D
        The container class to be initialized with the preprocessed data.
    data : array_like
        The raw data to be preprocessed.

    Attributes
    ----------
    container : Container2D
        The container that was initialized with the preprocessed data.
    """

    def __init__(self, container: Type["Container2D"], data: ArrayLike):
        x_raw, y_raw, data = self.prep(data)
        self.container = container(x_raw=x_raw, y_raw=y_raw, data=data)

    def __call__(self) -> Type["Container2D"]:
        """
        Return the container when the instance is called.

        Returns
        -------
        container : object
            The container that was initialized with the preprocessed data.

        Notes
        -----
        This method was implemented since this is a straightforward way to provide the prepped container object.
        """
        return self.container

    @abstractmethod
    def prep(self, data):
        """
        Process the provided data.

        This method should be implemented in subclasses to specify the preprocessing steps.

        Parameters
        ----------
        data : any
            The raw data to be preprocessed.

        Returns
        -------
        x_raw: np.ndarray
            1-dimensional numpy array with shape of (n, ).
        y_raw: np.ndarray
            1-dimensional numpy array with shape of (n, ).
        data: np.ndarray
            Structured array with proper column names.
        """
        pass


class ImageLoader(DataLoaderBase):
    def prep(self, data: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process the provided image data.

        Provided dimension of the image data should be 2 or 3. If the dimension is 2, then the image is assumed to be
        grayscale. If the dimension is 3, then the image is assumed to be multi-channel. The number of channels is not
        limited.

        Parameters
        ----------
        data : any
            The raw image data to be preprocessed.

        Returns
        -------
        x_serial, y_serial, prepped_data : tuple
            The preprocessed image data. x_serial and y_serial are the flattened arrays
            for the X and Y coordinates, and prepped_data is the serialized and reshaped image data.
        """
        x_serial, y_serial, image = image_notator(data)  # Get X and Y coordinates.
        if image.ndim == 3:
            prepped_data = image_serializer(image)
        elif image.ndim == 2:
            prepped_data = reshape_if_1dim(image_serializer(image))
        else:
            raise ValueError("Error: Image dimension error.")
        return x_serial.flatten(), y_serial.flatten(), prepped_data


class XYDLoader(DataLoaderBase):
    def prep(self, data: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process the provided XYD data.

        Parameters
        ----------
        data : array_like or structured array
            The raw XYD data to be preprocessed. The first and second columns should be
            x- and y-coordinates, and the third and subsequent columns should be data columns.

        Returns
        -------
        x_raw, y_raw, data : tuple
            The preprocessed XYD data. x_raw and y_raw are the x- and y-coordinates,
            and data contains the rest of the columns.
        """
        x_raw, y_raw, data = xyd_splitter(data)
        return x_raw, y_raw, data
