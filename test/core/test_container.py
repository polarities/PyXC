import numpy as np
import pytest

from pyxc.core.container import Container2D


# Return a prepped 'Container2D' object with sample x and y coordinates, and additional data in 'data' array.
def get_prepped_container():
    x_raw = np.array([1, 2, 3])
    y_raw = np.array([4, 5, 6])
    data = np.array([[7, 8], [9, 10], [11, 12]])
    default_columns_prefix = "Channel"
    return Container2D(x_raw, y_raw, data, default_columns_prefix)


# Test function to check if the 'Container2D' object is correctly initialized for an ideal case.
def test_init_works_as_expected_ideal_case():
    x_raw = np.array([1, 2, 3])
    y_raw = np.array([4, 5, 6])
    data = np.array([[7, 8], [9, 10], [11, 12]])
    default_columns_prefix = "Channel"

    # Test instantiation and data correctness
    container = Container2D(x_raw, y_raw, data, default_columns_prefix)
    assert np.array_equal(container["x_raw"], x_raw)
    assert np.array_equal(container["y_raw"], y_raw)
    assert np.all(np.isnan(container["x"]))
    assert np.all(np.isnan(container["y"]))
    assert np.array_equal(container["Channel_0"], data[:, 0])
    assert np.array_equal(container["Channel_1"], data[:, 1])


# Test function to check if 'Container2D' raises 'ValueError' when 'x_raw' and 'y_raw' arrays have different lengths.
def test_init_raises_ValueError_when_xy_raw_has_different_length():
    x_raw = np.array([1, 2, 3, 4])
    y_raw = np.array([4, 5, 6])
    data = np.array([[7, 8], [9, 10], [11, 12]])
    default_columns_prefix = "Channel"

    # Test instantiation and data correctness
    with pytest.raises(ValueError):
        container = Container2D(x_raw, y_raw, data, default_columns_prefix)


# Test function to check if 'Container2D' raises 'ValueError' when 'x_raw' or 'y_raw' arrays have more than one dimension.
def test_init_raises_ValueError_when_xy_raw_ndim_is_gt_1():
    x_raw = np.array(
        [
            [1, 2, 3],
        ]
    )
    y_raw = np.array(
        [
            [4, 5, 6],
        ]
    )
    data = np.array([[7, 8], [9, 10], [11, 12]])
    default_columns_prefix = "Channel"

    # Test instantiation and data correctness
    with pytest.raises(ValueError):
        container = Container2D(x_raw, y_raw, data, default_columns_prefix)


# Test function to check the behavior of the 'is_x_calibrated' property in the 'Container2D' class.
def test_is_x_calibrated():
    # Test when 'x' is not calibrated
    container = get_prepped_container()
    assert not container.is_x_calibrated

    # Test when 'x' is calibrated
    container["x"] = [1, 2, 3]
    assert container.is_x_calibrated


# Test function to check if setting raw x values correctly updates 'x_raw' in the 'Container2D' object.
def test_set_x_raw():
    container = get_prepped_container()
    # Test setting raw x values
    x_raw_new = [10, 20, 30]
    container.x_raw = x_raw_new
    assert np.array_equal(container["x_raw"], x_raw_new)


# Test function to check if setting raw y values correctly updates 'y_raw' in the 'Container2D' object.
def test_set_y_raw():
    container = get_prepped_container()
    # Test setting raw y values
    y_raw_new = [40, 50, 60]
    container.y_raw = y_raw_new
    assert np.array_equal(container["y_raw"], y_raw_new)

def test_add_column():
    container = get_prepped_container()

    # Test adding a column
    container_new = container.add_column("Channel_2", [13, 14, 15], dtype=int)
    assert np.array_equal(container_new["Channel_2"], [13, 14, 15])

    # Test the original container has not changed (KeyError)
    with pytest.raises(ValueError):
        container["Channel_2"]
