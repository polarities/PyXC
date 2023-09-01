import numpy as np
import pytest

from pyxc.transform.transform_base import MatrixTransformationBase


# Override the `_matrix_integrity_checker` method to always return `True`, enabling the testing of the `TransformationCommon` class using this custom behavior.
class ConcreteTransformationCommon(
    MatrixTransformationBase
):  # To test TransformationCommon class.
    def _matrix_integrity_checker(self, matrix):
        return True


# The `identity_matrix` fixture returns a 3x3 identity matrix, which can be used in tests to represent the transformation matrix of an identity transformation.
@pytest.fixture
def identity_matrix():
    return np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )


# Check the initialization of `ConcreteTransformationCommon` class with no parameters.
# It ensures that the class instance has the correct initial state and the identity matrix is set correctly.
def test_initialization_no_parameters(identity_matrix):
    t = ConcreteTransformationCommon()
    assert isinstance(t, MatrixTransformationBase)
    assert t._externally_set_transformation_matrix is None
    assert t.parent is None
    assert t.queue == []
    assert t.is_updated == False
    assert np.array_equal(t.identity, identity_matrix)


# Check the initialization of `ConcreteTransformationCommon` class with a transformation matrix provided as a parameter.
# It ensures that the provided matrix is correctly set as the externally set transformation matrix.
def test_initialization_with_transformation_matrix():
    matrix = np.array([[2, 0, 10], [0, 3, 40], [0, 0, 1]])
    t = ConcreteTransformationCommon(transformation_matrix=matrix)
    assert np.array_equal(t._externally_set_transformation_matrix, matrix)


# Check the `set_transformation_matrix` method of the `ConcreteTransformationCommon` class.
# It verifies that the transformation matrix can be correctly set using the method and that the `is_updated` flag is correctly set to True.
def test_set_transformation_matrix():
    t = ConcreteTransformationCommon()
    matrix = np.array([[2, 0, 10], [0, 3, 40], [0, 0, 1]])
    t.transformation_matrix = matrix
    assert np.array_equal(t._externally_set_transformation_matrix, matrix)
    assert t.is_updated


# Check that accessing `externally_set_transformation_matrix` property returns the identity matrix when no transformation matrix has been externally set.
def test_access_externally_set_transformation_matrix_when_not_set(identity_matrix):
    t = ConcreteTransformationCommon()
    assert np.array_equal(t.externally_set_transformation_matrix, identity_matrix)


# Check that accessing the `transformation_matrix` property returns the correct transformation matrix when there is no parent transformation and no queued transformations.
def test_access_transformation_matrix_no_parent_no_queue():
    t = ConcreteTransformationCommon()
    matrix = np.array([[2, 0, 10], [0, 3, 40], [0, 0, 1]])
    t.transformation_matrix = matrix
    assert np.array_equal(t.transformation_matrix, matrix)


# Check that accessing the `transformation_matrix` property returns the correct transformation matrix when there is a parent transformation and no queued transformations.
def test_access_transformation_matrix_no_queue_with_parent():
    parent_matrix = np.array([[2, 0, 10], [0, 3, 40], [0, 0, 1]])
    t1 = ConcreteTransformationCommon(transformation_matrix=parent_matrix)

    # Initialize second transformation matrix (at this moment no parent)
    child_matrix = np.array([[1, 0, 10], [0, 1, 5], [0, 0, 1]])
    t2 = ConcreteTransformationCommon(transformation_matrix=child_matrix)
    assert np.array_equal(t2.transformation_matrix, child_matrix)

    # Set parent
    t2.parent = t1
    assert not np.array_equal(t2.transformation_matrix, child_matrix)
    assert np.array_equal(t2.transformation_matrix, child_matrix @ parent_matrix)


# Check that accessing the `transformation_matrix` property returns the correct transformation matrix when there is a queue of transformations and no parent transformation.
def test_acess_transformation_matrix_with_queue_no_parent():
    ex_matrix = np.array([[3, 0, 0], [0, 4, 0], [0, 0, 1]])
    q1_matrix = np.array([[2, 0, 10], [0, 3, 40], [0, 0, 1]])
    q2_matrix = np.array([[1, 0, 10], [0, 1, 5], [0, 0, 1]])

    t1 = ConcreteTransformationCommon(transformation_matrix=ex_matrix)
    assert np.array_equal(t1.transformation_matrix, ex_matrix)

    t1.queue.append(q1_matrix)
    assert np.array_equal(t1.transformation_matrix, q1_matrix)

    t1.queue.append(q2_matrix)
    assert np.array_equal(t1.transformation_matrix, q2_matrix @ q1_matrix)


# Check that the child transformation object's parent and parent transformation matrix are set correctly when the parent is provided during initialization.
def test_set_parent():
    parent_matrix = np.array([[2, 0, 10], [0, 3, 40], [0, 0, 1]])
    child_matrix = np.array([[1, 0, 10], [0, 1, 5], [0, 0, 1]])
    t1 = ConcreteTransformationCommon(transformation_matrix=parent_matrix)
    t2 = ConcreteTransformationCommon(parent=t1, transformation_matrix=child_matrix)
    assert t2.parent == t1
    assert np.array_equal(t2.parent.transformation_matrix, parent_matrix)
