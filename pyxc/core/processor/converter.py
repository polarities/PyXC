from typing import Union

import numpy as np
from numpy.typing import ArrayLike


def iterable_to_numpy(array_like: ArrayLike) -> np.ndarray:
    """
    Return np.ndarray from the given `array_like`.

    Convert provided `array_like` to np.ndarray object if it is not a np.ndarray.

    Parameters
    ----------
    array_like : array_like
        Iterable to be converted to a `np.ndarray`.

    Returns
    -------
    np.ndarray
    """
    if not isinstance(array_like, np.ndarray):
        array_like = np.array(array_like)
    return array_like
