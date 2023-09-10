from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Type, Optional, NoReturn
from warnings import warn

import numpy as np
from numpy.lib import recfunctions as rfn
from scipy.spatial import cKDTree
from tqdm import tqdm

from pyxc.core.container import Container2D
from pyxc.core.layer_manager import LayerRegistry
from pyxc.core.loader import DataLoaderBase
from pyxc.core.processor.reducer import Reducer
from pyxc.transform.affine2d import Affine2D
from pyxc.transform.homography import Homography
from pyxc.transform.transform_base import TransformationBase

LAYERS = LayerRegistry()


class Layer(object):
    """
    Manage a data container and transformation object.

    Parameters
    ----------
        data : array_like
            Data to be stored in the layer.
        dataloader : Callable
            Function to load the data into the container.
        transformer : Homography | Affine2D
            Transformation class for the layer.
        container : Container2D, optional
            Container class to hold the data. Default is Container2D.
        name : str, optional
            Name for the layer. Default is None.
        parent : Layer, optional
            Parent Layer to inherit transformations from. Default is None.
        layer_manager : LayerRegistry, optional
            LayerRegistry instance to manage the layer. Default is None.
        **kwargs : dict, optional
            Additional parameters to be passed to the container.

    Attributes
    ----------
    is_transformed : bool
        Indicates whether the layer has been calibrated.
    manager : LayerRegistry, optional
        A reference to the layer registry.
    transformer : Homography | Affine2D
        An instance of a transformation object.
    parent : Layer, optional
        A reference to the parent Layer, if one exists.
    container : Container2D
        Container for the data.

    Methods
    -------
    apply_transformation(self)
        Apply the transformation to the raw x and y coordinates in the container.
    get_x(self, interval=None)
        Return the x-coordinates, applying transformation if not already done.
    get_y(self, interval=None)
        Return the y-coordinates, applying transformation if not already done.
    get_xy(self, interval=None)
        Return the x and y-coordinates, applying transformation if not already done.
    get_x_raw(self, interval=None)
        Return the raw x-coordinates.
    get_y_raw(self, interval=None)
        Return the raw y-coordinates.
    get_xy_raw(self, interval=None)
        Return the raw x and y-coordinates.
    get_data(self, interval=None)
        Return the data at a specified interval.
    query(self, x, y, cutoff=1, output_number=1, reducer=None, idx=None)
        Return the data at a query point (x, y) within the cutoff distance.
    execute_queries(self, xs, ys, cutoff=1, output_number=1, reducer=None, max_workers=-1)
        Execute multiple queries in parallel using lists of x and y coordinates.
    _integrity_check(self)
        Check the integrity of the layer by verifying the container's validity.
    set_parent(self, layer_object)
        Set a parent for this layer.
    """

    def __init__(
        self,
        data,
        dataloader: Type["DataLoaderBase"],
        transformer: Type["TransformationBase"],
        container: Type["Container2D"] = Container2D,
        name: Optional[str] = None,
        parent: Optional["Layer"] = None,
        layer_manager: Optional["LayerRegistry"] = LAYERS,
        **kwargs,
    ):
        self.is_transformed = False
        self.kd_tree = None

        # Register as a layer
        self.manager = layer_manager
        if self.manager is not None:  # If layer manager is given:
            try:  # Then register
                self.manager.register(self, desired_name=name)
            except (ValueError, KeyError) as err:  # Destroy the object when failed.
                warn(f"Exception raised: {repr(err)}")
                warn(
                    f"Self-destructing created layer object. Please try to create a new one."
                )
                self.__del__()

        # Initialise required things
        if callable(transformer):  # Little messy. TODO: Better design.
            self.transformer: Type[
                "TransformationBase"
            ] = transformer()  # Transformation object
        else:
            self.transformer: Type["TransformationBase"] = transformer

        self.parent: None | Layer = (
            parent  # Just initialize the property. Will be set later.
        )
        if parent is not None:
            self.set_parent(parent)

        # Load data
        self.container: Type["Container2D"] = dataloader(
            data=data, container=container
        )()  # return value of loader is container object
        self.is_transformed = False

    def __del__(self):
        """
        Destruct a Layer object.

        When a Layer object is deleted, it is also deregistered from the LayerRegistry.
        """

        if self.manager is not None:
            self.manager.request_deletion(self)

    def reset_tree(self) -> NoReturn:
        """Reset the cKDTree in the current object."""
        if self.kd_tree is not None:
            self.kd_tree = None

    def build_tree(self) -> NoReturn:
        """Build a cKDTree for the nearest neighbour calculations."""
        if self.kd_tree is None:
            self.kd_tree = cKDTree(
                np.column_stack([self.container["x"], self.container["y"]])
            )

    def get_layer_manager(self) -> LayerRegistry:
        return self.manager

    def apply_transformation(self) -> NoReturn:
        """
        Apply the transformation to the raw x and y coordinates in the container and mark the layer as transformed.
        """
        xc, yc = self.transformer.apply_transformation(
            self.container.x_raw, self.container.y_raw
        )
        self.container.x = xc
        self.container.y = yc
        self.is_transformed = True

    @property
    def x_raw(self) -> np.ndarray:
        return self.container.x_raw

    @x_raw.setter
    def x_raw(self, value):
        if self.container.is_x_raw_exists:
            warn(
                "You are trying to override the container `x_raw` column. "
                "The KD Tree and the container `x` values are also reset."
            )
        # Raw X replaced -> X need to be recalculated.
        self.container.x_raw = value
        self.container.reset_calibration_x()
        self.is_transformed = False

        # X need to be recalculated -> New tree required.
        self.reset_tree()

    @property
    def y_raw(self):
        return self.container.y_raw

    @y_raw.setter
    def y_raw(self, value):
        if self.container.is_y_raw_exists:
            warn(
                "You are trying to override the container `x_raw` column. "
                "The KD Tree and the container `x` values are also reset."
            )
        # Raw X replaced -> X needs to be recalculated.
        self.container.y_raw = value
        self.container.reset_calibration_y()
        self.is_transformed = False

        # X needs to be recalculated -> New tree required.
        self.reset_tree()

    @property
    def xy_raw(self):
        return self.container.x_raw, self.container.y_raw

    @property
    def x(self):
        if self.is_transformed is False:
            self.apply_transformation()
        return self.container.x

    @property
    def y(self):
        if self.is_transformed is False:
            self.apply_transformation()
        return self.container.y

    @property
    def xy(self):
        return self.container.x, self.container.y

    def get_x(self, interval: Optional[int] = None) -> np.ndarray:
        if self.is_transformed is False:
            self.apply_transformation()
        if interval is not None:
            return self.container.x[::interval]
        else:
            return self.container.x

    def get_y(self, interval: Optional[int] = None) -> np.ndarray:
        if self.is_transformed is False:
            self.apply_transformation()
        if interval is not None:
            return self.container.y[::interval]
        else:
            return self.container.y

    def get_xy(self, interval: Optional[int] = None) -> np.ndarray:
        if self.is_transformed is False:
            self.apply_transformation()
        if interval is not None:
            return self.container.x[::interval], self.container.y[::interval]
        else:
            return self.container.x, self.container.y

    def get_x_raw(self, interval: Optional[int] = None) -> np.ndarray:
        if interval is not None:
            return self.container.x_raw[::interval]
        else:
            return self.container.x_raw

    def get_y_raw(self, interval: Optional[int] = None) -> np.ndarray:
        if interval is not None:
            return self.container.y_raw[::interval]
        else:
            return self.container.y_raw

    def get_xy_raw(self, interval: Optional[int] = None) -> np.ndarray:
        if interval is not None:
            return self.container.x_raw[::interval], self.container.y_raw[::interval]
        else:
            return self.container.x_raw, self.container.y_raw

    def query(
        self,
        x: float,
        y: float,
        cutoff: float = 1,
        output_number: int = 1,
        reducer: Reducer = None,
        idx: int | None = None,
    ) -> np.ndarray:
        """
        Execute a single query with given x and y coordinates.

        Parameters
        ----------
        x : float
            The x-coordinate for the query.
        y : float
            The y-coordinate for the query.
        cutoff : float, optional
            The maximum distance from the query point to consider data.
        output_number : int, optional
            The number of closest points to return for the query.
        reducer : Reducer, optional
            Reducer object to reduce multiple query results into a single result.
        idx : int | None, optional
            The index of the query. This option is used when executing multiple queries in parallel by the `execute_queries`
            method. This parameter is obsolete for single query execution.

        Returns
        -------
        np.ndarray
            An array of query results, sorted by the original index of the query.
        """
        if self.is_transformed is False:
            self.apply_transformation()

        # Initialize the cKD tree for the efficient NN-search.
        self.build_tree()

        # kNN-search
        dists, indices = self.kd_tree.query(
            [x, y], k=output_number, distance_upper_bound=cutoff
        )
        dists = np.array(dists).flatten()
        indices = np.array(indices).flatten()[dists != np.inf]
        dists = dists[dists != np.inf]
        if len(dists) > 0:
            base = self.container[indices]
        else:
            base = np.empty(0, dtype=self.container.dtype)
            warn("Couldn't find the matching point. Please ignore rows containing NaN.")
        # Form a result.

        result = rfn.append_fields(
            base=base,
            names=(
                "query_index",
                "distance",
                "x-coordinates",
                "y-coordinates",
            ),
            data=(
                np.array([idx if idx is not None else 0] * len(dists), dtype=int),
                dists,
                np.array(
                    [
                        x,
                    ]
                    * len(dists)
                ),
                np.array(
                    [
                        y,
                    ]
                    * len(dists)
                ),
            ),
            usemask=False,
        )

        if len(dists) == 0:
            result = np.empty(0, dtype=result.dtype)

        if reducer is not None:
            return reducer.reduce(result)
        else:
            return result

    def execute_queries(
        self,
        xs: list[float],
        ys: list[float],
        cutoff: float = 1,
        output_number: int = 1,
        reducer: Optional[Reducer] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Execute multiple queries in parallel, given lists of x and y coordinates.

        Parameters
        ----------
        xs : list[float]
            List of x coordinates for queries.
        ys : list[float]
            List of y coordinates for queries.
        cutoff : float, optional
            The maximum distance from the query point to consider data.
        output_number : int, optional
            The number of closest points to return for each query.
        reducer : Reducer, optional
            Reducer object to reduce multiple query results into a single result.
        **kwargs : dict, optional
            Additional parameters to be passed to the ThreadPoolExecutor.

        Returns
        -------
        np.ndarray
            An array of query results, sorted by the original index of the query.
        """

        if len(xs) != len(ys):
            raise ValueError(
                "Error: The length of xs must be the same as the length of ys"
            )

        if output_number != 1 and reducer is None:
            raise ValueError(
                "Error: Output number must be 1 (nearest-neighbour) or reducer should be specified."
            )

        results = []

        with ThreadPoolExecutor(**kwargs) as executor:
            print("Maximum worker: ", executor._max_workers)
            tasks = {
                executor.submit(
                    self.query, x, y, cutoff, output_number, reducer, idx
                ): (x, y)
                for idx, (x, y) in enumerate(zip(xs, ys))
            }

            for future in tqdm(
                as_completed(tasks), total=len(tasks), desc="Executing queries"
            ):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Query failed for {tasks[future]}. Reason: {str(e)}")
        if results:  # Check if results list is not empty
            results = np.concatenate(results)
        return results[np.argsort(results["query_index"])]

    def _integrity_check(self) -> bool:
        """
        Check if the layer is intact by checking if the container is valid.

        Returns
        -------
        bool
            True if all elements in the container are valid, False otherwise.
        """

        if all(self.container):
            return True
        else:
            return False

    def set_parent(self, layer_object: Type["Layer"]) -> NoReturn:
        """
        Set a parent for this layer. The parent's transformation is also set to this layer's transformer's parent.

        Parameters
        ----------
        layer_object : Layer
            The parent Layer to be set.

        Raises
        ------
        TypeError
            If layer_object is not an instance of Layer.
        """
        if isinstance(layer_object, Layer):
            self.parent = layer_object
            self.transformer.parent = layer_object.transformer
            self.is_transformed = False
        else:
            raise TypeError(
                "Error: The given object is not a supported type. Please provide a layer object."
            )

    def set_transformation_matrix(self, transformation_matrix) -> NoReturn:
        """
        Set the transformation matrix for the layer.

        Parameters
        ----------
        transformation_matrix : np.ndarray
            The transformation matrix to be set.

        Notes
        -----
        ..deprecated:: 1.0.0
            This method will be removed in PyXC 2.0.0.
        """
        self.transformer.transformation_matrix = transformation_matrix
        self.container.reset_calibration()
        self.is_transformed = False
