from .transform_base import TransformationBase
import numpy as np
from abc import abstractmethod


class VectorisedParametricTransformationBase(TransformationBase):
    @abstractmethod
    def eq(self, x: np.ndarray, y: np.ndarray, *args, **kwargs):
        pass

    def apply_transformation(self, x: np.ndarray, y: np.ndarray):
        return self.eq(x, y)


class QuadracticSurfaceFit(VectorisedParametricTransformationBase):
    """Quadratic surface fit transformation. Not fully implemented yet."""

    def eq(self, x, y, *args, **kwargs):
        a1 = kwargs.get("ax", 0)
        b1 = kwargs.get("bx", 0)
        c1 = kwargs.get("cx", 0)
        d1 = kwargs.get("dx", 0)
        e1 = kwargs.get("ex", 0)
        f1 = kwargs.get("fx", 0)

        a2 = kwargs.get("ay", 0)
        b2 = kwargs.get("by", 0)
        c2 = kwargs.get("cy", 0)
        d2 = kwargs.get("dy", 0)
        e2 = kwargs.get("ey", 0)
        f2 = kwargs.get("fy", 0)

        return (
            a1 * x**2 + b1 * y**2 + c1 * x + d1 * y + e1 + f1,
            a2 * x**2 + b2 * y**2 + c2 * x + d2 * y + e2 + f2,
        )
