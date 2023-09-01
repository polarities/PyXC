import numpy as np
from numpy.lib import recfunctions as rfn
from numpy.typing import ArrayLike
from ..core.processor.arrays import reshape_if_1dim
from ..core.processor.checker import is_structured_array
from typing import NoReturn


class Container2D(np.ndarray):
    """
    A container class that extends the numpy ndarray to handle x and y data points.
    """

    def __new__(
        cls,
        x_raw: ArrayLike,
        y_raw: ArrayLike,
        data: ArrayLike,
        default_columns_prefix: str = "Channel",
        *args,
        **kwargs,
    ):
        if np.ndim(x_raw) != 1 or np.ndim(y_raw) != 1:
            raise ValueError(
                "Error:`x_raw` or `y_raw` properties must be in 1-dimensional iterable."
            )
        if len(x_raw) != len(y_raw):
            raise ValueError("Error: `x_raw` or `y_raw` must be in same length.")

        dtype = [
            ("row", np.uint16),
            ("x", "float32"),
            ("y", "float32"),
            ("x_raw", "float32"),
            ("y_raw", "float32"),
        ]

        xy = np.array(
            list(
                zip(
                    [x for x in range(len(x_raw))],
                    tmp := [None for _ in x_raw],
                    tmp,
                    x_raw,
                    y_raw,
                )
            ),
            dtype=dtype,
        )

        if not is_structured_array(
            data
        ):  # If data is not structured array then asign default column name.
            data = reshape_if_1dim(data)
            data = rfn.unstructured_to_structured(
                data,
                np.dtype(
                    [
                        (f"{default_columns_prefix}_{i}", data.dtype.type)
                        for i in range(data.shape[1])
                    ]
                ),
            )
        xyd = rfn.merge_arrays([xy, data], flatten=True, usemask=False)
        return xyd.view(cls)

    @property
    def is_x_calibrated(self) -> bool:
        return not (all(np.isnan(self["x"])))

    @property
    def is_y_calibrated(self) -> bool:
        return not (all(np.isnan(self["y"])))

    @property
    def is_calibrated(self) -> bool:
        return True if self.is_x_calibrated and self.is_y_calibrated else False

    @property
    def is_x_raw_exists(self) -> bool:
        return not (all(np.isnan(self["x_raw"])))

    @property
    def is_y_raw_exists(self) -> bool:
        return not (all(np.isnan(self["y_raw"])))

    @property
    def is_xy_raw_exist(self) -> bool:
        return True if self.is_x_raw_exists and self.is_y_raw_exists else False

    # Raw X and Y property handlers
    @property
    def x_raw(self) -> np.ndarray:
        """Get the raw x values."""
        if self["x_raw"] is None:
            raise ValueError("Error: Origin x values were not set.")
        return self["x_raw"]

    @x_raw.setter
    def x_raw(self, value) -> NoReturn:
        """
        Set the raw x values.

        Parameters
        ----------
        value : np.ndarray
            Original x values.
        """
        self["x_raw"] = value

    @property
    def y_raw(self) -> np.ndarray:
        """Get the raw y values."""
        if self["y_raw"] is None:
            raise ValueError("Error: Origin y values were not set.")
        return self["y_raw"]

    @y_raw.setter
    def y_raw(self, value) -> NoReturn:
        """
        Set the raw y values.

        Parameters
        ----------
        value : np.ndarray
            Original y values.
        """
        self["y_raw"] = value

    # Calibrated X and Y property handlers
    @property
    def x(self) -> np.ndarray:
        """Get the calibrated x values, falling back on raw if not available."""
        return self["x"] if self.is_x_calibrated else self["x_raw"]

    @x.setter
    def x(self, value) -> NoReturn:
        """
        Set the calibrated x values.

        Parameters
        ----------
        value : np.ndarray
            Calibrated x values.

        Notes
        -----
        This method updates the calibrated x values. This method is intended to be used internally.
        """
        self["x"] = value

    @property
    def y(self) -> np.ndarray:
        """Get the calibrated y values, falling back on raw if not available."""
        return self["y"] if self.is_y_calibrated else self["y_raw"]

    @y.setter
    def y(self, value) -> NoReturn:
        """
        Set the calibrated y values.

        Parameters
        ----------
        value : np.ndarray
            Calibrated y values.

        Notes
        -----
        This method updates the calibrated y values. This method is intended to be used internally.
        """
        self["y"] = value

    # Getting and setting methods
    def get_x(self) -> np.ndarray:
        """Return calibrated x or raw x if calibrated does not exist."""
        return self.x

    def get_y(self) -> np.ndarray:
        """Return calibrated y or raw y if calibrated does not exist."""
        return self.y

    def get_xy(self) -> np.ndarray:
        """Return a tuple of (x, y)."""
        return self.get_x(), self.get_y()

    def add_column(self, name, data, dtype) -> np.ndarray:
        """
        Add a new column to the structured array.

        Parameters
        ----------
        name : str
            Name of the new column.
        data : np.ndarray
            Data to be added.
        dtype : int, float, complex, str, or object
            Data type of the new column.

        Returns
        -------
        np.ndarray

        Notes
        -----
        This operation does not perform in-place modification. You need to manually assign the returned values to the container.
        """
        new_dtype = self.dtype.descr + [(name, dtype)]
        new_data = np.zeros(self.shape, dtype=new_dtype)

        for field in self.dtype.names:
            new_data[field] = self[field]

        new_data[name] = data
        return new_data.view(type(self))

    def reset_calibration_x(self) -> NoReturn:
        self["x"] = [None for _ in self["x_raw"]]

    def reset_calibration_y(self) -> NoReturn:
        self["y"] = [None for _ in self["y_raw"]]

    def reset_calibration(self) -> NoReturn:
        self.reset_calibration_x()
        self.reset_calibration_y()
