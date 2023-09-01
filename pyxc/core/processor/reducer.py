import numpy as np
from numpy.typing import ArrayLike


class Reducer(list):
    """
    The Reducer class is designed to perform reductions on structured arrays.

    This class is particularly useful when working with large amounts of data and a subset of information or
    simplified representation is needed. The `reduce` method transforms the structured array to a simpler
    form based on the specified functions/callables added to the Reducer list. An instance of the Reducer
    class is a list of functions or tuples with two elements: a function and a list of columns from the
    array that the function should be applied to. The functions in the list are applied to the structured
    array in the order they are added to the list.

    Methods
    -------
    reduce(array: ArrayLike) -> np.ndarray
        Apply the functions in the Reducer list to the specified array and return the reduced array.

    Notes
    -----
    The structured array must be a 2D structured NumPy array. The structured array is a powerful tool
    that allows you to manipulate data with different types and sizes. It's like a 2D table with each
    column possibly of a different type.

    The output of the reduce method is a structured array with new columns corresponding to the results
    of the applied functions. For example, if the function 'np.mean' is applied to the columns 'x' and 'y'
    of the array, the output will include the columns 'x_mean' and 'y_mean'.
    """

    def reduce(self, array: np.ndarray) -> np.ndarray:
        final_dtype = list(
            [
                ("count", int),
                ("query_index", int),
                ("x-coordinates", float),
                ("y-coordinates", float),
                ("avg_x", float),
                ("avg_y", float),
            ]
        )
        final_array = list(
            [
                len(array),
                array["query_index"][0],
                array["x-coordinates"][0],
                array["y-coordinates"][0],
                np.mean(array["x"]),
                np.mean(array["y"]),
            ]
        )

        for idx, action in enumerate(self):
            if callable(action):
                act = action
                cols = array.dtype.names
            elif len(action) == 2:
                act, cols = action
                for col in cols:
                    if col not in array.dtype.names:
                        raise ValueError(
                            f"Error: A given column name {col} not in array."
                        )
            else:
                raise ValueError(
                    "Error: Only two elements are allowed for each reducer element."
                )
            suffix = act.__name__
            reduced_value = [act(array[column]) for column in cols]
            reduced_value_dtype = [
                (f"{name}_{suffix}", array.dtype[name])
                for name in array.dtype.names
                if name in cols
            ]
            final_dtype += reduced_value_dtype
            final_array += reduced_value
        return np.array(
            [
                tuple(final_array),
            ],
            np.dtype(final_dtype),
        )
