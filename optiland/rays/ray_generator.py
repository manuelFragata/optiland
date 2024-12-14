"""Ray Generator

This module contains the RayGenerator class, which is used to generate rays
for tracing through an optical system.

Kramer Harrison, 2024
"""

from abc import ABC, abstractmethod
import pickle
import hashlib
import numpy as np
from optiland.rays.real_rays import RealRays
from optiland.rays.polarized_rays import PolarizedRays
from optiland.rays.ray_cache import RayCache
from optiland.rays.ray_aiming import RayAimerFactory


def get_ray_starting_z_offset(optic, EPD=None):
    """
    Calculate the starting ray z-coordinate offset for systems with an
    object at infinity. This is relative to the first surface of the optic.

    This method chooses a starting point that is equivalent to the entrance
    pupil diameter of the optic.

    Args:
        optic (Optic): The optical system.
        EPD (float): The entrance pupil diameter of the optic. If None, the
            EPD is calculated from the paraxial data.

    Returns:
        float: The z-coordinate offset relative to the first surface.
    """
    z = optic.surface_group.positions[1:-1]
    if EPD is None:
        offset = optic.paraxial.EPD()
    else:
        offset = EPD
    return offset - np.min(z)


def get_ray_origins_finite(optic, field, pupil):
    """
    Get ray origin points for a finite object.

    Args:
        Hx (float): Normalized x field coordinate.
        Hy (float): Normalized y field coordinate.
        Px (float or np.ndarray): x-coordinate of the pupil point.
        Py (float or np.ndarray): y-coordinate of the pupil point.
        vx (float): Vignetting factor in the x-direction.
        vy (float): Vignetting factor in the y-direction.

    Returns:
        tuple: A tuple containing the x, y, and z coordinates of the
            ray starting position.
    """
    field_x, field_y = field
    Px, Py = pupil

    obj = optic.object_surface

    # TODO: use object coordinate system
    x = field_x
    y = field_y
    z = obj.geometry.sag(x, y) + obj.geometry.cs.z

    x0 = np.full_like(Px, x)
    y0 = np.full_like(Px, y)
    z0 = np.full_like(Px, z)

    return x0, y0, z0


def get_ray_origins_infinite(optic, field, pupil, vx, vy):
    """
    Get ray origin points for an infinit object.

    Args:
        optic (Optic): The optical system.
        field (tuple): A tuple containing the normalized x and y field
            coordinates.
        pupil (tuple): A tuple containing the x and y pupil coordinates.
        vx (float): Vignetting factor in the x-direction.
        vy (float): Vignetting factor in the y-direction.

    Returns:
        tuple: A tuple containing the x, y, and z coordinates of the
            ray starting position.
    """
    field_x, field_y = field
    Px, Py = pupil

    EPL = optic.paraxial.EPL()
    EPD = optic.paraxial.EPD()

    offset = get_ray_starting_z_offset(optic, EPD)

    # x, y, z positions of ray starting points
    x = np.tan(np.radians(field_x)) * (offset + EPL)
    y = -np.tan(np.radians(field_y)) * (offset + EPL)
    z = optic.surface_group.positions[1] - offset

    x0 = Px * EPD / 2 * vx + x
    y0 = Py * EPD / 2 * vy + y
    z0 = np.full_like(Px, z)

    return x0, y0, z0


class FieldCalculator(ABC):

    def __init__(self, optic):
        self.optic = optic

    def generate_rays(self, Hx, Hy, Px, Py, wavelength):
        """
        Generates rays for tracing based on the given parameters.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            Px (float or np.ndarray): x-coordinate of the pupil point.
            Py (float or np.ndarray): y-coordinate of the pupil point.
            wavelength (float): Wavelength of the rays.

        Returns:
            RealRays: RealRays object containing the generated rays.
        """
        self._validate()

        vx, vy = 1 - np.array(self.optic.fields.get_vig_factor(Hx, Hy))
        x0, y0, z0 = self._get_ray_origins(Hx, Hy, Px, Py, vx, vy)

        if self.optic.obj_space_telecentric:
            ap_type = self.optic.aperture.ap_type

            if ap_type == 'objectNA':
                sin = self.optic.aperture.value
            elif ap_type == 'object_cone_angle':
                sin = np.sin(np.radians(self.optic.aperture.value))
            else:
                raise ValueError(f'Aperture type {ap_type} may not be used '
                                 'with telecentric object space.')

            z = np.sqrt(1 - sin**2) / sin + z0
            z1 = np.full_like(Px, z)
            x1 = Px * vx + x0
            y1 = Py * vy + y0
        else:
            EPL = self.optic.paraxial.EPL()
            EPD = self.optic.paraxial.EPD()

            x1 = Px * EPD * vx / 2
            y1 = Py * EPD * vy / 2
            z1 = np.full_like(Px, EPL)

        return self._build_ray_instance(x0, y0, z0, x1, y1, z1, wavelength)

    @abstractmethod
    def _get_ray_origins(self, Hx, Hy, Px, Py, vx, vy):
        """
        Calculate the initial positions for rays originating at the object.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            Px (float or np.ndarray): x-coordinate of the pupil point.
            Py (float or np.ndarray): y-coordinate of the pupil point.
            vx (float): Vignetting factor in the x-direction.
            vy (float): Vignetting factor in the y-direction.

        Returns:
            tuple: A tuple containing the x, y, and z coordinates of the
                object position.
        """
        pass  # pragma: no cover

    @abstractmethod
    def _validate(self):
        """
        Validate the field calculator settings.
        """
        pass  # pragma: no cover

    def _build_ray_instance(self, x0, y0, z0, x1, y1, z1, wavelength):
        """
        Build a RealRays or PolarizedRays instance based on a set of ray
        starting and ending points, wavelength, and polarization setting.

        Args:
            x0 (float or np.ndarray): x-coordinate of the starting point.
            y0 (float or np.ndarray): y-coordinate of the starting point.
            z0 (float or np.ndarray): z-coordinate of the starting point.
            x1 (float or np.ndarray): x-coordinate of the ending point.
            y1 (float or np.ndarray): y-coordinate of the ending point.
            z1 (float or np.ndarray): z-coordinate of the ending point.
            wavelength (float): Wavelength of the rays.

        Returns:
            RealRays or PolarizedRays: The appropriate ray instance.
        """
        mag = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)
        L = (x1 - x0) / mag
        M = (y1 - y0) / mag
        N = (z1 - z0) / mag

        x0 = np.full_like(x1, x0)
        y0 = np.full_like(x1, y0)
        z0 = np.full_like(x1, z0)

        intensity = np.ones_like(x1)
        wavelength = np.ones_like(x1) * wavelength

        if self.optic.polarization == 'ignore':
            if self.optic.surface_group.uses_polarization:
                raise ValueError('Polarization must be set when surfaces have '
                                 'polarization-dependent coatings.')
            return RealRays(x0, y0, z0, L, M, N, intensity, wavelength)
        else:
            return PolarizedRays(x0, y0, z0, L, M, N, intensity, wavelength)


class AngleFieldCalculator(FieldCalculator):

    def __init__(self, optic):
        super().__init__(optic)

    def _get_ray_origins(self, Hx, Hy, Px, Py, vx, vy):
        """
        Calculate the initial positions for rays originating in object space.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            Px (float or np.ndarray): x-coordinate of the pupil point.
            Py (float or np.ndarray): y-coordinate of the pupil point.
            vx (float): Vignetting factor in the x-direction.
            vy (float): Vignetting factor in the y-direction.

        Returns:
            tuple: A tuple containing the x, y, and z coordinates of the
                object position.
        """
        obj = self.optic.object_surface

        max_field = self.optic.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy
        field = (field_x, field_y)
        pupil = (Px, Py)

        if obj.is_infinite:
            return get_ray_origins_infinite(self.optic, field, pupil, vx, vy)

        else:
            EPL = self.optic.paraxial.EPL()
            z = self.optic.surface_group.positions[0]
            x = np.tan(np.radians(field_x)) * (EPL - z)
            y = -np.tan(np.radians(field_y)) * (EPL - z)

            x0 = np.full_like(Px, x)
            y0 = np.full_like(Px, y)
            z0 = np.full_like(Px, z)

            return x0, y0, z0

    def _validate(self):
        if self.optic.obj_space_telecentric:
            raise ValueError('Object space cannot be telecentric for a '
                             'field type of "angle".')


class ObjectHeightFieldCalculator(FieldCalculator):

    def __init__(self, optic):
        super().__init__(optic)

    def _get_ray_origins(self, Hx, Hy, Px, Py, vx, vy):
        """
        Calculate the initial positions for rays originating at the object.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            Px (float or np.ndarray): x-coordinate of the pupil point.
            Py (float or np.ndarray): y-coordinate of the pupil point.
            vx (float): Vignetting factor in the x-direction.
            vy (float): Vignetting factor in the y-direction.

        Returns:
            tuple: A tuple containing the x, y, and z coordinates of the
                object position.
        """
        max_field = self.optic.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy
        return get_ray_origins_finite(self.optic, (field_x, field_y), (Px, Py))

    def _validate(self):
        # Check if the object space is infinite
        infinite = self.optic.object_surface.is_infinite
        if infinite:
            raise ValueError('Object space cannot be infinite for a '
                             'field type of "object_height".')

        # Object space cannot be telecentric for EPD or imageFNO apertures
        telecentric = self.optic.obj_space_telecentric
        ap_type = self.optic.aperture.ap_type
        if telecentric and ap_type in ['EPD', 'imageFNO']:
            raise ValueError(f'Aperture type cannot be "{ap_type}" for'
                             ' telecentric object space.')


class ParaxialImageHeightFieldCalculator(FieldCalculator):

    def __init__(self, optic):
        super().__init__(optic)

    def _get_ray_origins(self, Hx, Hy, Px, Py, vx, vy):
        """
        Calculate the initial positions for rays originating at the object.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            Px (float or np.ndarray): x-coordinate of the pupil point.
            Py (float or np.ndarray): y-coordinate of the pupil point.
            vx (float): Vignetting factor in the x-direction.
            vy (float): Vignetting factor in the y-direction.

        Returns:
            tuple: A tuple containing the x, y, and z coordinates of the
                object position.
        """
        y_img, H_img = self._generate_mapping()
        Hx = np.interp(Hx, H_img, y_img)
        Hy = np.interp(Hy, H_img, y_img)

        obj = self.optic.object_surface

        max_field = self.optic.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy
        field = (field_x, field_y)
        pupil = (Px, Py)

        if obj.is_infinite:
            return get_ray_origins_infinite(self.optic, field, pupil, vx, vy)

        else:
            EPL = self.optic.paraxial.EPL()
            z = self.optic.surface_group.positions[0]
            x = np.tan(np.radians(field_x)) * (EPL - z)
            y = -np.tan(np.radians(field_y)) * (EPL - z)

            x0 = np.full_like(Px, x)
            y0 = np.full_like(Px, y)
            z0 = np.full_like(Px, z)

            return x0, y0, z0

    def _validate(self):
        pass

    def _generate_mapping(self):
        """
        Generate a mapping between field coordinates and paraxial image height.

        Returns:
            tuple: A tuple containing paraxial image height and field
                coordinates
        """
        num = 32
        Hy = np.linspace(0, 1, num)
        Py = np.zeros(num)
        wavelength = self.optic.primary_wavelength

        rays = self.optic.paraxial.trace(Hy, Py, wavelength)
        return rays.y, Hy


class FieldCalculatorFactory:

    @staticmethod
    def create(optic):
        if optic.field_type == 'angle':
            return AngleFieldCalculator(optic)
        elif optic.field_type == 'object_height':
            return ObjectHeightFieldCalculator(optic)
        elif optic.field_type == 'paraxial_image_height':
            return ParaxialImageHeightFieldCalculator(optic)
        else:
            raise ValueError('Invalid field type: {}'.format(optic.field_type))


class RayGenerator:

    def __init__(self, optic):
        self.optic = optic
        self.aiming_type = None
        self.ray_aimer = None
        self.field_calculator = FieldCalculatorFactory.create(optic)
        self.cache = None

    def generate_rays(self, Hx, Hy, Px, Py, wavelength):
        if self.ray_aimer:
            # Generate a unique key for the ray cache
            key = self._generate_cache_key(Hx, Hy, Px, Py, wavelength)

            # Check if the rays are already in the cache
            rays = self.cache.get_rays(key)
            if rays:
                return rays

            # Generate the initial rays for aiming
            aim_key = (Hx, Hy, Px, Py, wavelength)
            initial_rays = self.ray_aimer.aim_rays(aim_key)

            # If the initial rays are not found, generate new rays
            if not initial_rays:
                initial_rays = self.field_calculator.generate_rays(
                    Hx, Hy, Px, Py, wavelength)

            # Generate the final rays
            rays = self.ray_aimer.generate_rays(Hx, Hy, Px, Py, initial_rays)

            # Add the rays to the cache
            self.cache.add_rays(key, rays)

        else:
            rays = self.field_calculator.generate_rays(
                Hx, Hy, Px, Py, wavelength)

        return rays

    def set_ray_aiming(self, aiming_type):
        self.aiming_type = aiming_type
        if aiming_type:
            self.cache = RayCache()
            self.ray_aimer = RayAimerFactory.create(self.optic, aiming_type)
        else:
            self.cache = None
            self.ray_aimer = None

    def _generate_object_hash(self):
        # Serialize the system object
        serialized_sys = pickle.dumps(self.optic)

        # Create a hash of the serialized data
        hash_obj = hashlib.sha256(serialized_sys)

        # Return the hexadecimal representation of the hash
        return hash_obj.hexdigest()

    def _generate_cache_key(self, Hx, Hy, Px, Py, wavelength):
        system_hash = self._generate_object_hash()
        return (system_hash, self.aiming_type, Hx, Hy, Px, Py, wavelength)
