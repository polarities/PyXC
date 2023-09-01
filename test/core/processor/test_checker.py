import numpy as np

from pyxc.core.processor.checker import is_structured_array


def test_is_structured_array():
    # +
    structured_dtype = np.dtype([("x", np.float64), ("y", np.int32)])
    structured_array = np.array([(1.5, 6), (2.5, 7)], dtype=structured_dtype)
    assert is_structured_array(structured_array) == True

    # -
    non_structured_array1 = np.array([1, 2, 3, 4, 5])
    assert is_structured_array(non_structured_array1) == False

    non_structured_array2 = np.array([[1, 2], [3, 4]])
    assert is_structured_array(non_structured_array2) == False

    non_structured_array3 = np.array([{"a": 1}, {"b": 2}])
    assert is_structured_array(non_structured_array3) == False
