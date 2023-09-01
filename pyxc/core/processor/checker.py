import numpy as np


def is_structured_array(array: np.ndarray) -> bool:
    """
    Check whether the provided array is structured.

    Parameters
    ----------
    array : np.ndarray
        Array to be checked.

    Returns
    -------
    bool
        True if the provided array is a structured array. False when the provided array is a non-structured array.

    Notes
    -----
    `NumPy documentation <https://numpy.org/doc/stable/user/basics.rec.html#manipulating-and-displaying-structured-datatypes>`_
    """

    if (array.dtype.names is None) and (array.dtype.fields is None):
        return False
    else:
        return True
