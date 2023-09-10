import numpy as np
import pytest

from pyxc.core.container import Container2D
from pyxc.core.layer import Layer
from pyxc.core.layer_manager import LayerRegistry
from pyxc.core.loader import ImageLoader
from pyxc.core.processor.reducer import Reducer
from pyxc.transform.affine2d import Affine2D
from pyxc.transform.homography import Homography

# Pytest fixture providing a `Layer` object with sample image data for testing.
IMAGE_SIZE = 20
# Prep sample image
sample_data = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
sample_data[:, :, 0] = np.arange(IMAGE_SIZE)
sample_data[:, :, 1] = np.arange(IMAGE_SIZE)[::-1]
sample_data[:, :, 2] = 100

reducer = Reducer([(np.mean, ("Channel_0",))])
LAYERS = LayerRegistry()


# Pytest fixtures for testing `Layer` initialization and registration.
# The `Layer` object is created with sample image data, an `ImageLoader` dataloader, a `Homography` transformer,
# a `Container2D` container, and is managed by the `LayerRegistry` in `LAYERS`.
@pytest.fixture
def layer():
    return Layer(
        data=sample_data,
        dataloader=ImageLoader,
        transformer=Homography,
        container=Container2D,
        layer_manager=LAYERS,
    )


# Return a `Reducer` object configured to calculate the mean of the 'Channel_0' data.
@pytest.fixture
def reducer():
    return Reducer([(np.mean, ("Channel_0",))])


# Check if the `Layer` object is properly initialized and not transformed.
def test_layer_initialization(layer):
    assert isinstance(layer, Layer)
    assert not layer.is_transformed

def test_layer_initialization_with_parent(layer):
    L2_with_parent = Layer(
        data=sample_data,
        dataloader=ImageLoader,
        transformer=Affine2D,
        container=Container2D,
        parent=layer,
    )

    assert L2_with_parent.parent == layer
    assert L2_with_parent.transformer.parent == layer.transformer

# Verify if the `Layer` object is successfully registered in `LAYERS`.
def test_layer_registration(layer):
    assert layer in LAYERS.values()


# Check 'is_transformed' status and data equality before and after applying transformations.
def test_layer_transformation(layer):
    # Newly configured layer should not be transformed.
    assert layer.is_transformed == False

    # After applying transformation without specifying transform matrix, value should be identical.
    layer.apply_transformation()
    assert layer.is_transformed  # Should be True now.
    assert np.array_equal(layer.get_x(), layer.get_x_raw())  # Also should be equal.
    assert np.array_equal(layer.get_y(), layer.get_y_raw())  # Also should be equal.

    # Setting transformation matrix
    layer.set_transformation_matrix(
        np.array(
            [
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 1],
            ]
        )
    )
    assert layer.is_transformed == False  # Should be False now.
    layer.apply_transformation()
    assert layer.is_transformed  # Should be True now.
    assert np.array_equal(layer.get_x(), layer.get_x_raw() * 2)  # Scaled for 2 times.
    assert np.array_equal(layer.get_y(), layer.get_y_raw() * 2)  # Scaled for 2 times.


# Check if query returns the expected result based on the provided reducer.
def test_query(layer, reducer):
    layer.apply_transformation()
    result = layer.query(5, 6, reducer=reducer)
    assert len(result) == 1
    assert result["query_index"] == 0


# Verify if query returns the expected result when the query point is outside the cutoff range.
def test_query_out_of_cutoff_range(layer):
    layer.apply_transformation()
    result = layer.query(IMAGE_SIZE + 10, IMAGE_SIZE + 10, cutoff=3, output_number=1)
    assert len(result) == 0


# Check if executing multiple queries without a reducer returns the expected number of results.
def test_execute_queries_without_reducer_output_1(layer):
    layer.apply_transformation()
    results = layer.execute_queries([5, 9], [6, 10], 2, 1)
    assert len(results) == 2


# Check if executing multiple queries with a reducer returns the expected number of results.
def test_execute_queries_with_reducer_output_4(layer, reducer):
    layer.apply_transformation()
    results = layer.execute_queries([10, 15, 20], [6, 8, 10], 2, 4, reducer=reducer)
    assert len(results) == 3


# Verify if setting a parent layer correctly updates the child layer and its transformer.
def test_set_parent():
    parent_layer = Layer(
        data=sample_data,
        dataloader=ImageLoader,
        transformer=Homography,
        container=Container2D,
        layer_manager=LAYERS,
    )
    child_layer = Layer(
        data=sample_data,
        dataloader=ImageLoader,
        transformer=Affine2D,
        container=Container2D,
        layer_manager=LAYERS,
    )
    child_layer.set_parent(parent_layer)
    assert child_layer.parent == parent_layer
    assert child_layer.transformer.parent == parent_layer.transformer


def test_query_query_index(layer):
    result = layer.query(
        x=5, y=5, cutoff=5, output_number=5
    )  # If not specified query_index should be 0 for all rows.
    assert np.count_nonzero(result["query_index"] == 0) == 5

    result = layer.query(x=5, y=5, cutoff=5, output_number=5, idx=1)  # If specified
    assert np.count_nonzero(result["query_index"] == 1) == 5
