import operator
from collections import OrderedDict
from typing import NoReturn, Type, Optional

# Type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyxc.core.layer import Layer


class LayerRegistry(OrderedDict):
    """Manages layer objects in an ordered dictionary for easy access and operation."""

    def __init__(self):
        """
        Initialize the LayerRegistry.
        """
        super(LayerRegistry, self).__init__()
        self.name_increment: int = (
            -1
        )  # After returns it is not possible to increment this value. So starting from -1.
        self.name_prefix: str = "layer"

    def register(self, layer_object: Type["Layer"], desired_name: Optional[str] = None):
        """
        Register a Layer object in the LayerRegistry.

        The registered layer can be accessed iteratively and be manipulated using other LayerRegistry methods.

        Parameters
        ----------
        layer_object : Layer
            Layer object to be registered.
        desired_name : str, optional
            Desired name of the layer. If not specified, the default name will be assigned.

        Raises
        ------
        KeyError
            If the layer is already registered.
        ValueError
            If the desired name is already occupied.
        """
        for (
            name,
            layer,
        ) in self.items():  # Check given layer object is already registered.
            if layer_object is layer:  # If already registered,
                raise KeyError(
                    f"Error: A layer object you have tried to register is already registered. "
                    f"Registered as: {name} @ {id(layer)}"
                )
        if (desired_name is not None) and (
            desired_name in self.keys()
        ):  # If desired_name is already taken.
            raise ValueError(
                f"Error: Sorry! Your desired name {desired_name} is already occupied."
            )
        if desired_name is None:  # If desired_name is not specified, then assign name.
            desired_name = self._request_name_automatic()
        self.update({desired_name: layer_object})  # Register the layer_object.

    def broadcast(self, action: str, *args, **kwargs) -> NoReturn:
        """
        Perform a specified action on all registered layers.

        This function can be used to apply any method supported by the Layer class to all registered layers.

        Parameters
        ----------
        action : str
            Action to be performed on all registered layers.
        *args
            Arguments to be passed on the action.
        **kwargs
            Keyword arguments to be passed on the action.
        """
        for name, layer in self.items():
            operator.attrgetter(action)(layer)(*args, **kwargs)

    def scan(self, *desired_properties) -> dict:
        """
        Get specified properties from all registered Layer objects.

        This function can be used to access any property supported by the Layer class from all registered layers.

        Parameters
        ----------
        *desired_properties : str
            Properties to be accessed from all registered layers.
        """
        intermediate_dictionary_outer = dict()
        for (
            key,
            value,
        ) in self.items():  # Loop over the dictionary to take a value from each layer
            intermediate_dictionary_inner = dict()
            for property in desired_properties:  # Loop over the property
                intermediate_dictionary_inner.update(
                    {property: operator.attrgetter(property)(value)}
                )
            intermediate_dictionary_outer[key] = intermediate_dictionary_inner
        return intermediate_dictionary_outer

    @staticmethod
    def status_update(during_operation: str, after_completion: str = "idle"):
        """
        Update the status of the LayerRegistry. Currently not implemented.

        Parameters
        ----------
        during_operation : str
            The status during an operation.
        after_completion : str
            The status after the operation is completed.
        """
        raise NotImplementedError

    def request_deletion(self, layer_object: "Layer") -> NoReturn:
        """
        Delete a registered layer from the LayerRegistry.

        Parameters
        ----------
        layer_object : Layer
            Layer object to be deleted.

        Raises
        ------
        KeyError
            If the layer is not found in the LayerRegistry.
        """
        for key, value in self.items():
            if value is layer_object:
                del self[key]
                break

    def _request_name_automatic(self, prefix: str = "layer") -> str:
        """
        Generate a default layer name.

        If no prefix is specified, the default prefix 'layer' is used.

        Parameters
        ----------
        prefix : str
            Prefix of the layer name.
        """
        if prefix == self.name_prefix:  # If prefix is consistent, then increment.
            self.name_increment += 1
        else:  # Else, reset suffix number.
            self.name_prefix = prefix
            self.name_increment = 0

        candidate = (
            f"{self.name_prefix}_{self.name_increment}"  # Make initial candidate.
        )

        while candidate in self.keys():  # If initial candidate is duplicated name.
            self.name_increment += 1
            candidate = f"{self.name_prefix}_{self.name_increment}"

        return candidate
