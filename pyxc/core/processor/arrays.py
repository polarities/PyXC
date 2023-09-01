from typing import Sequence, Optional

import numpy as np
from numpy.lib.recfunctions import (
    structured_to_unstructured,
    unstructured_to_structured,
)
from numpy.typing import ArrayLike

from pyxc.core.processor.checker import is_structured_array
from pyxc.core.processor.converter import iterable_to_numpy


def reshape_if_1dim(array: np.ndarray, is_structured=False) -> np.ndarray:
    """
    Reshape the provided array if it's one-dimensional.

    If the shape of the provided array is (n, ), then it will be reshaped to (n, 1). If the array is two-dimensional or
    is a structured array, it will be returned as is. Otherwise, a ValueError will be raised.

    Parameters
    ----------
    array : np.ndarray
        The input array to be reshaped.
    is_structured : bool, default=False
        A boolean flag indicating whether the input array is structured.

    Returns
    -------
    np.ndarray
        The reshaped array.

    Raises
    ------
    ValueError
        If the dimensions of the input array are either too low or too high.
    """
    if array.ndim == 1:
        return array.reshape(-1, 1)
    elif array.ndim == 2 or is_structured:
        return array
    else:
        raise ValueError(
            f"Error: given array cannot be handled in this class. Dimension is too low or too high.\n"
            f"Given dimension: {array.ndim}"
        )


def image_notator(
    image: ArrayLike, channel: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Label X- and Y-indices of the provided image.

    Parameters
    ----------
    image : array_like or np.ndarray
        A 2D or 3D array-like object representing the image.
        For 3D arrays, the last dimension should be the channel.

    channel : int, optional
        Specifies the channel to be considered if the input image is a 3D array.
        If None (default), all channels are considered.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing:
        - X-indices of the image as a ndarray
        - Y-indices of the image as a ndarray
        - The supplied image or its specific channel as a ndarray

    Raises
    ------
    ValueError
        If the dimension of the input `image` is not 2D or 3D.
    """

    image = iterable_to_numpy(image)
    if image.ndim == 3:  # If color image.
        y, x = np.indices(
            np.shape(image[..., 0])
        )  # TODO: Check whether (y, x, _) is faster.
    elif image.ndim == 2:
        y, x = np.indices(np.shape(image))
    else:
        raise ValueError("Error: Image dimension is not right.")

    if channel is None:
        return x, y, image
    else:
        return x, y, image[channel]


def image_serializer(image: ArrayLike) -> tuple[np.ndarray, ...]:
    """
    Flatten image to tuple(s).

    Return flattened image from the supplied image.

    Parameters
    ----------
    image : array_like
        2-dimensional or 3-dimensional array-like image.

    Returns
    -------
    tuple[np.ndarray, ...]
        Flattened image. Each row contains image information.
    """
    image = iterable_to_numpy(image)
    if image.ndim == 3:
        height, width, channels = image.shape
        flattened_image = image.reshape(height * width, channels)
        return flattened_image
    else:
        return image.flatten()


def get_structured_array_field_names_from_indices(
    dtype_object: np.dtype, indices: Sequence[int]
) -> tuple[str, ...]:
    """
    Return field names from the provided `dtype_object` based on the integer indices.

    Parameters
    ----------
    dtype_object : np.dtype
        A `np.dtype` object of the structured array.
    indices : Sequence
        A sequence of the desired index or indices.

    Returns
    -------
    tuple[str, ...]
    """
    names: tuple = dtype_object.names
    return tuple(names[key] for key in indices)


def xy2matrix(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Return (n, 3) shaped array after receiving two (n,) shaped arrays.

    Parameters
    ----------
    x : array_like
        NumPy array contains x-coordinate values.
    y : array_like
        NumPy array contains y-coordinate values.

    Returns
    -------
    np.ndarray
    """
    z = np.ones(len(x))
    return np.stack((x, y, z), axis=0)


def matrix2xy(matrix_like_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the first (x) and second (y) columns from the provided (n, 3) shaped array.

    Parameters
    ----------
    matrix_like_array : np.ndarray
        (n, 3) shaped array. First and second columns are x and y coordinates.

    Returns
    -------
    np.ndarray
        x-coordinates
    np.ndarray
        y-coordinates
    """
    return (
        np.array(matrix_like_array[0]).flatten(),
        np.array(matrix_like_array[1]).flatten(),
    )


def structured_array_query(
    structured_array: np.ndarray, indices: Sequence[int]
) -> np.ndarray:
    """
    Return a structured array based on the given integer indices.

    Get a subset of the structured array based on given indices. Usually, column slicing of the structured array is
    performed using key values. This function provides structured array column slicing by integer column indices.

    Parameters
    ----------
    structured_array : np.ndarray
        Structured array.
    indices : Sequence[int, ...]
        A sequence of integer column indices.

    Returns
    -------
    np.ndarray
        Structured array.
    """
    dtype_obj = structured_array.dtype
    unstr = structured_to_unstructured(structured_array)
    sorted_array = unstr[..., indices]
    new_dtype_obj = dtype_obj[
        list(get_structured_array_field_names_from_indices(dtype_obj, indices))
    ]
    return unstructured_to_structured(sorted_array, new_dtype_obj)


def column_parser(
    array: ArrayLike, format_string: str, return_unspecified=True
) -> np.ndarray:
    """
    Re-order the provided array_like object with respect to the format string.

    Parameters
    ----------
    array : array_like
        2-dimensional array or structured array.
    format_string : str
        A Format string. See Notes section.
    return_unspecified : bool, default=True
        If `False`, the unspecified portion of columns will *NOT* be returned together. See Notes.

    Returns
    -------
    np.ndarray
        Re-ordered np.ndarray respective with the provided format string.

    Raises
    ------
    ValueError
        When the provided format string or the provided array-like object is not right.

    Notes
    -----
    A format string key-letter is `x`, `y`, and `_`. Each letter corresponds to x-coordinates, y-coordinates, and columns
    to be ignored. All other characters than `x`, `y`, and `_` will be treated as valid columns.

    If `return_unspecified` is False, then unspecified columns won't be returned together. For example, when the
    format string is given by `yxd__dd` and the provided array is a structured array with column names of
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), then this function returns columns (2, 1, 3, 6, 7, 8, 9, 10) by default.
    However, when `return_unspecified` is False, then this function returns (2, 1, 3, 6, 7) only.
    """

    format_string = format_string.lower()  # Convert string to the list.
    num_x = list(format_string).count("x")
    num_y = list(format_string).count("y")

    array = iterable_to_numpy(
        array
    )  # Check the given array is numpy array. If not, convert to np.ndarray.

    if num_x != 1:  # a number of the x-coordinates column should be 1.
        raise ValueError("Error:The number of the x-column is (more / less) than 1.")
    if num_y != 1:  # a number of the y-coordinates column should be 1.
        raise ValueError("Error: The number of the y-column is (more / less) than 1.")

    try:  # If 2-dimensional array (not a structured array).
        rows, cols = np.shape(array)
        is_structured = False
    except (
        ValueError
    ):  # If structured array, then there is a possibility of only rows dimension is exists.
        rows = np.shape(array)
        cols = len(array.dtype.names)
        is_structured = True
        if (
            cols < 2
        ):  # At least two columns are required to describe x and y coordinates
            raise ValueError(
                "Error: The number of columns is not enough. At least 2 columns are required to define 'xy'"
            )

    if len(format_string) > cols:
        raise ValueError(
            "Error: The length of the provided string is larger than the number of columns of the provided array."
        )

    # Finding indices of the columns.
    # x = x-coordinates, y = y-coordinates, _ = ignored, others = data.
    # if "_xy_ddddf" specified, then x: 1, y: 2, d: 4, 5, 6, 7, 8
    x_col_location = list(format_string).index("x")
    y_col_location = list(format_string).index("y")
    ignored_cols_location = list(
        i for i, x in enumerate(list(format_string)) if x == "_"
    )
    data_cols_location = list(
        set(range(cols))
        - set(ignored_cols_location)
        - {x_col_location}  # {i,} is a set literal
        - {y_col_location}  # {i,} is a set literal
    )

    field_indices = tuple(
        [x_col_location] + [y_col_location] + data_cols_location
    )  # (x, y, d, d, ...)

    # if number of columns is 5 but 'xy_d' specified,
    if return_unspecified is False:  # when return_unspecified is False, return xy_d.
        specified_index_until = len(format_string) - 1
        field_indices = tuple(x for x in field_indices if x <= specified_index_until)
        if len(field_indices) < 2:  # At least X, Y should be returned.
            raise ValueError("Error: Parser option string is not properly specified.")
    else:  # when return_unspecified is True, return xy_dd.
        pass

    if not is_structured:  # if array is not a structured array
        return array[..., field_indices]
    else:  # if array is a structured array
        str_fields = get_structured_array_field_names_from_indices(
            array.dtype, field_indices
        )
        return structured_array_query(array, field_indices)


def xyd_splitter(array: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a structured array into three separate arrays.

    Parameters
    ----------
    array : array_like
        Array-like object to be split. If it is a structured array, then the number of columns should be greater than or equal
        to 3.

    Returns
    -------
    np.ndarray
        X-coordinates. Unstructured array. Shape is (n, )
    np.ndarray
        Y-coordinates. Unstructured array. Shape is (n, )
    np.ndarray
        Data array. If the initially given `array` and number of the columns is greater than or equal to 4 in the structured
        array, then return a structured array. Else, return a 2-dimensional unstructured array.

    Raises
    ------
    ValueError
        If the input array has less than three columns.
    """
    array = iterable_to_numpy(array)  # Convert to NumPy array.
    is_structured = is_structured_array(
        array
    )  # Check if the provided array is a structured array.

    if (
        is_structured is True
    ):  # If it is structured, the number of columns is equal to the number of records.
        cols = len(array.dtype.names)
    else:
        _, cols = np.shape(
            array
        )  # If not structured, the number of columns can be obtained using the `np.shape()` method.

    if cols <= 3:  # 3 is minimum number.
        raise ValueError(
            "Error: The number of columns is not sufficient. To return X, Y, and Data columns at least "
            "3 columns are required. Please check the array you have provided."
        )

    # Return split array. x, y, and rest.
    if is_structured is True:
        key_x = array.dtype.names[0]
        key_y = array.dtype.names[1]
        key_data = get_structured_array_field_names_from_indices(
            array.dtype, range(2, cols)
        )
        return array[key_x], array[key_y], structured_array_query(array, range(2, cols))
    else:
        return (
            array[..., 0],
            array[..., 1],
            array[..., tuple(x for x in range(2, cols))],
        )
