"""Optiland Optic Module

This is the core module of Optiland, which provides the class to define
optical systems.

Kramer Harrison, 2024
"""
import numpy as np
from optiland.fields import Field, FieldGroup
from optiland.surfaces import SurfaceGroup, ObjectSurface
from optiland.wavelength import WavelengthGroup
from optiland.paraxial import Paraxial
from optiland.aberrations import Aberrations
from optiland.aperture import Aperture
from optiland.rays import RealRays
from optiland.distribution import create_distribution
from optiland.geometries import Plane, StandardGeometry
from optiland.materials import IdealMaterial
from optiland.visualization import LensViewer, LensViewer3D


class Optic:
    """
    The Optic class represents an optical system.

    Attributes:
        aperture (Aperture): The aperture of the optical system.
        field_type (str): The type of field used in the optical system.
        surface_group (SurfaceGroup): The group of surfaces in the optical
            system.
        fields (FieldGroup): The group of fields in the optical system.
        wavelengths (WavelengthGroup): The group of wavelengths in the optical
            system.
        paraxial (Paraxial): The paraxial analysis helper class for the
            optical system.
        aberrations (Aberrations): The aberrations analysis helper class for
            the optical system.
    """

    def __init__(self):
        self.aperture = None
        self.field_type = None

        self.surface_group = SurfaceGroup()
        self.fields = FieldGroup()
        self.wavelengths = WavelengthGroup()

        self.paraxial = Paraxial(self)
        self.aberrations = Aberrations(self)

    @property
    def primary_wavelength(self):
        """float: the primary wavelength in microns"""
        return self.wavelengths.primary_wavelength.value

    @property
    def object_surface(self):
        """Surface: the object surface instance"""
        for surface in self.surface_group.surfaces:
            if isinstance(surface, ObjectSurface):
                return surface
        return None

    @property
    def image_surface(self):
        """Surface: the image surface instance"""
        return self.surface_group.surfaces[-1]

    @property
    def total_track(self):
        """float: the total track length of the system"""
        z = self.surface_group.positions[1:-1]
        return np.max(z) - np.min(z)

    def add_surface(self, new_surface=None, surface_type='standard',
                    index=None, is_stop=False, material='air', thickness=0,
                    **kwargs):
        """
        Adds a new surface to the optic.

        Args:
            new_surface (Surface, optional): The new surface to add. If not
                provided, a new surface will be created based on the other
                arguments.
            surface_type (str, optional): The type of surface to create.
            index (int, optional): The index at which to insert the new
                surface. If not provided, the surface will be appended to the
                end of the list.
            is_stop (bool, optional): Indicates if the surface is the aperture.
            material (str, optional): The material of the surface.
                Default is 'air'.
            thickness (float, optional): The thickness of the surface.
                Default is 0.
            **kwargs: Additional keyword arguments for surface-specific
                parameters such as radius, conic, dx, dy, rx, ry, aperture.

        Raises:
            ValueError: If index is not provided when defining a new surface.
        """
        self.surface_group.add_surface(
            new_surface=new_surface, surface_type=surface_type, index=index,
            is_stop=is_stop, material=material, thickness=thickness, **kwargs
            )

    def add_field(self, y, x=0.0, vx=0.0, vy=0.0):
        """
        Add a field to the optical system.

        Args:
            y (float): The y-coordinate of the field.
            x (float, optional): The x-coordinate of the field.
                Defaults to 0.0.
            vx (float, optional): The x-component of the field's vignetting
                factor. Defaults to 0.0.
            vy (float, optional): The y-component of the field's vignetting
                factor. Defaults to 0.0.
        """
        new_field = Field(self.field_type, x, y, vx, vy)
        self.fields.add_field(new_field)

    def add_wavelength(self, value, is_primary=False, unit='um'):
        """
        Add a wavelength to the optical system.

        Args:
            value (float): The value of the wavelength.
            is_primary (bool, optional): Whether the wavelength is the primary
                wavelength. Defaults to False.
            unit (str, optional): The unit of the wavelength. Defaults to 'um'.
        """
        self.wavelengths.add_wavelength(value, is_primary, unit)

    def set_aperture(self, aperture_type, value):
        """
        Set the aperture of the optical system.

        Args:
            aperture_type (str): The type of the aperture.
            value (float): The value of the aperture.
        """
        self.aperture = Aperture(aperture_type, value)

    def set_field_type(self, field_type):
        """
        Set the type of field used in the optical system.

        Args:
            field_type (str): The type of field.
        """
        self.field_type = field_type

    def set_radius(self, value, surface_number):
        """
        Set the radius of curvature of a surface.

        Args:
            value (float): The value of the radius.
            surface_number (int): The index of the surface.
        """
        surface = self.surface_group.surfaces[surface_number]

        # change geometry from plane to standard
        if isinstance(surface.geometry, Plane):
            cs = surface.geometry.cs
            new_geometry = StandardGeometry(cs, radius=value, conic=0)
            surface.geometry = new_geometry
        else:
            surface.geometry.radius = value

    def set_conic(self, value, surface_number):
        """
        Set the conic constant of a surface.

        Args:
            value (float): The value of the conic constant.
            surface_number (int): The index of the surface.
        """
        surface = self.surface_group.surfaces[surface_number]
        surface.geometry.k = value

    def set_thickness(self, value, surface_number):
        """
        Set the thickness of a surface.

        Args:
            value (float): The value of the thickness.
            surface_number (int): The index of the surface.
        """
        positions = self.surface_group.positions
        delta_t = value - positions[surface_number+1] + \
            positions[surface_number]
        positions[surface_number+1:] += delta_t
        positions -= positions[1]  # force surface 1 to be at zero
        for k, surface in enumerate(self.surface_group.surfaces):
            surface.geometry.cs.z = positions[k]

    def set_index(self, value, surface_number):
        """
        Set the index of refraction of a surface.

        Args:
            value (float): The value of the index of refraction.
            surface_number (int): The index of the surface.
        """
        surface = self.surface_group.surfaces[surface_number]
        new_material = IdealMaterial(n=value, k=0)
        surface.material_post = new_material

        surface_post = self.surface_group.surfaces[surface_number+1]
        surface_post.material_pre = new_material

    def set_asphere_coeff(self, value, surface_number, aspher_coeff_idx):
        """
        Set the asphere coefficient on a surface

        Args:
            value (float): The value of aspheric coefficient
            surface_number (int): The index of the surface.
            aspher_coeff_idx (int): index of the aspheric coefficient on the
                surface
        """
        surface = self.surface_group.surfaces[surface_number]
        surface.geometry.c[aspher_coeff_idx] = value

    def draw(self, fields='all', wavelengths='primary', num_rays=3,
             figsize=(10, 4)):
        """
        Draw a 2D representation of the optical system.

        Args:
            fields (str or list, optional): The fields to be displayed.
                Defaults to 'all'.
            wavelengths (str or list, optional): The wavelengths to be
                displayed. Defaults to 'primary'.
            num_rays (int, optional): The number of rays to be traced for each
                field and wavelength. Defaults to 3.
            figsize (tuple, optional): The size of the figure. Defaults to
                (10, 4).
        """
        viewer = LensViewer(self)
        viewer.view(fields, wavelengths, num_rays, distribution='line_y',
                    figsize=figsize)

    def draw3D(self, fields='all', wavelengths='primary', num_rays=2,
               figsize=(1200, 800)):
        """
        Draw a 3D representation of the optical system.

        Args:
            fields (str or list, optional): The fields to be displayed.
                Defaults to 'all'.
            wavelengths (str or list, optional): The wavelengths to be
                displayed. Defaults to 'primary'.
            num_rays (int, optional): The number of rays to be traced for each
                field and wavelength. Defaults to 2.
            figsize (tuple, optional): The size of the figure. Defaults to
                (1200, 800).
        """
        viewer = LensViewer3D(self)
        viewer.view(fields, wavelengths, num_rays,
                    distribution='hexapolar', figsize=figsize)

    def reset(self):
        """
        Reset the optical system to its initial state.
        """
        self.aperture = None
        self.field_type = None

        self.surface_group = SurfaceGroup()
        self.fields = FieldGroup()
        self.wavelengths = WavelengthGroup()

        self.paraxial = Paraxial(self)
        self.aberrations = Aberrations(self)

    def n(self, wavelength='primary'):
        """
        Get the refractive indices of the surfaces.

        Args:
            wavelength (float or str, optional): The wavelength for which to
                calculate the refractive indices. Defaults to 'primary'.

        Returns:
            numpy.ndarray: The refractive indices of the surfaces.
        """
        if wavelength == 'primary':
            wavelength = self.primary_wavelength
        n = []
        for surface in self.surface_group.surfaces:
            n.append(surface.material_post.n(wavelength))
        return np.array(n)

    def update_paraxial(self):
        """
        Update the semi-aperture of the surfaces based on the paraxial
        analysis.
        """
        ya, _ = self.paraxial.marginal_ray()
        yb, _ = self.paraxial.chief_ray()
        ya = np.abs(np.ravel(ya))
        yb = np.abs(np.ravel(yb))
        for k, surface in enumerate(self.surface_group.surfaces):
            surface.set_semi_aperture(r_max=ya[k]+yb[k])

    def image_solve(self):
        """Update the image position such that the marginal ray crosses the
        optical axis at the image location."""
        ya, ua = self.paraxial.marginal_ray()
        self.surface_group.surfaces[-1].geometry.cs.z -= ya[-1] / ua[-1]

    def trace(self, Hx, Hy, wavelength, num_rays=100,
              distribution='hexapolar'):
        """
        Trace a distribution of rays through the optical system.

        Args:
            Hx (float or numpy.ndarray): The normalized x field coordinate.
            Hy (float or numpy.ndarray): The normalized y field coordinate.
            wavelength (float): The wavelength of the rays.
            num_rays (int, optional): The number of rays to be traced. Defaults
                to 100.
            distribution (str or Distribution, optional): The distribution of
                the rays. Defaults to 'hexapolar'.
        """
        EPL = self.paraxial.EPL()
        EPD = self.paraxial.EPD()

        vx, vy = self.fields.get_vig_factor(Hx, Hy)

        if isinstance(distribution, str):
            distribution = create_distribution(distribution)
            distribution.generate_points(num_rays, vx, vy)
        x1 = distribution.x * EPD / 2
        y1 = distribution.y * EPD / 2
        z1 = np.ones_like(x1) * EPL

        rays = self._generate_rays(Hx, Hy, x1, y1, z1, wavelength, EPL)
        self.surface_group.trace(rays)

        return rays

    def trace_generic(self, Hx, Hy, Px, Py, wavelength):
        """
        Trace generic rays through the optical system.

        Args:
            Hx (float or numpy.ndarray): The normalized x field coordinate.
            Hy (float or numpy.ndarray): The normalized y field coordinate.
            Px (float or numpy.ndarray): The normalized x pupil coordinate.
            Py (float or numpy.ndarray): The normalized y pupil coordinate
            wavelength (float): The wavelength of the rays.
        """
        EPL = self.paraxial.EPL()
        EPD = self.paraxial.EPD()

        vx, vy = self.fields.get_vig_factor(Hx, Hy)

        x1 = Px * EPD / 2 * (1 - vx)
        y1 = Py * EPD / 2 * (1 - vy)

        # assure all variables are arrays of the same size
        max_size = max([np.size(arr) for arr in [x1, y1, Hx, Hy]])
        x1, y1, Hx, Hy = [
            np.full(max_size, value) if isinstance(value, (float, int))
            else value if isinstance(value, np.ndarray)
            else None
            for value in [x1, y1, Hx, Hy]
        ]

        z1 = np.ones_like(x1) * EPL

        rays = self._generate_rays(Hx, Hy, x1, y1, z1, wavelength, EPL)
        self.surface_group.trace(rays)

    def _generate_rays(self, Hx, Hy, x1, y1, z1, wavelength, EPL):
        """
        Generates rays for tracing based on the given parameters.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            x1 (float or np.ndarray): x-coordinate of the target point.
            y1 (float or np.ndarray): y-coordinate of the target point.
            z1 (float or np.ndarray): z-coordinate of the target point.
            wavelength (float): Wavelength of the rays.
            EPL (float): Entrance pupil position with respect to first surface.

        Returns:
            RealRays: RealRays object containing the generated rays.
        """
        x0, y0, z0 = self._get_object_position(Hx, Hy, x1, y1, EPL)

        mag = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)
        L = (x1 - x0) / mag
        M = (y1 - y0) / mag
        N = (z1 - z0) / mag

        x0 = np.ones_like(x1) * x0
        y0 = np.ones_like(x1) * y0
        z0 = np.ones_like(x1) * z0

        energy = np.ones_like(x1)
        wavelength = np.ones_like(x1) * wavelength

        return RealRays(x0, y0, z0, L, M, N, energy, wavelength)

    def _get_object_position(self, Hx, Hy, x1, y1, EPL):
        """
        Calculate the position of the object based on the given parameters.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            x1 (float or np.ndarray): x-coordinate of the target point.
            y1 (float or np.ndarray): y-coordinate of the target point.
            EPL (float): Entrance pupil position with respect to first surface.

        Returns:
            tuple: A tuple containing the x, y, and z coordinates of the
                object position.

        Raises:
            ValueError: If the field type is "object_height" for an object at
                infinity.

        """
        obj = self.object_surface
        max_field = self.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy
        if obj.is_infinite:
            if self.field_type == 'object_height':
                raise ValueError('''Field type cannot be "object_height" for an
                                 object at infinity.''')

            # start rays just before left-most surface (1/7th of total track)
            z = self.surface_group.positions[1:-1]
            offset = self.total_track / 7 - np.min(z)

            # x, y, z positions of ray starting points
            x = np.tan(np.radians(field_x)) * (offset + EPL)
            y = -np.tan(np.radians(field_y)) * (offset + EPL)
            z = self.surface_group.positions[1] - offset

            x0 = x1 + x
            y0 = y1 + y
            z0 = np.ones_like(x1) * z
        else:
            if self.field_type == 'object_height':
                x = field_x
                y = -field_y
                z = obj.geometry.sag(x, y) + obj.geometry.cs.z

                x0 = np.ones_like(x1) * x
                y0 = np.ones_like(x1) * y
                z0 = np.ones_like(x1) * z

            elif self.field_type == 'angle':
                x = np.tan(np.radians(field_x))
                y = -np.tan(np.radians(field_y))
                z = self.surface_group.positions[0]

                x0 = x1 + x
                y0 = y1 + y
                z0 = np.ones_like(x1) * z

        return x0, y0, z0
