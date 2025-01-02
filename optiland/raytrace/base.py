"""Base class for ray tracing through an optical system.

Kramer Harrison, 2025
"""
from abc import ABC, abstractmethod


class BaseRayTracer(ABC):
    """Base class for ray tracing through an optical system.

    Args:
        surface_group (SurfaceGroup): The surface group to trace rays through.
    """
    def __init__(self, surface_group):
        self.surface_group = surface_group

    @abstractmethod
    def trace(self, rays):
        """Trace rays through the optical system.

        Args:
            rays (BaseRays): The rays to trace through the optical system.
        """
        pass
