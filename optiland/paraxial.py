"""Paraxial Module

This module provides various functionalities for the computation of paraxial
properties of lens systems.

Kramer Harrison, 2024
"""
import optiland.backend as be
from optiland.raytrace import ParaxialRayTracer


class Paraxial:
    """
    A class representing a paraxial optical system.

    This class provides methods to calculate various properties of the optical
    system, such as focal lengths, entrance pupil location, exit pupil
    location, entrance pupil diameter, exit pupil diameter, image-space
    F-number, magnification, and more.

    Args:
        optic (Optic): The optical system to analyze.

    Attributes:
        optic (Optic): The optical system being analyzed.
        surfaces (SurfaceGroup): The surface group of the optical system.
    """

    def __init__(self, optic):
        self.optic = optic
        self.surfaces = self.optic.surface_group
        self.tracer = ParaxialRayTracer(optic)

    def trace(self, Hy, Py, wavelength):
        """
        Trace paraxial ray through the optical system based on specified field
        and pupil coordinates.

        Args:
            Hy (float): Normalized field coordinate.
            Py (float): Normalized pupil coordinate.
            wavelength (float): Wavelength of the light.
        """
        return self.tracer.trace(Hy, Py, wavelength)

    def f1(self):
        """Calculate the front focal length

        Returns:
            float: front focal length
        """
        # start tracing 1 lens unit before first surface
        z_start = -1
        wavelength = self.optic.primary_wavelength
        y, u = self.tracer.trace_generic(1.0, 0.0, z_start, wavelength,
                                         reverse=True)
        f1 = y[0] / u[-1]
        return f1[0]

    def f2(self):
        """Calculate the focal length

        Returns:
            float: back focal length
        """
        # start tracing 1 lens unit before first surface
        z_start = self.surfaces.positions[1] - 1
        wavelength = self.optic.primary_wavelength
        y, u = self.tracer.trace_generic(1.0, 0.0, z_start[0], wavelength)
        f2 = -y[0] / u[-1]
        return be.abs(f2[0])

    def F1(self):
        """Calculate the front focal point location

        Returns:
            float: front focal point location
        """
        # start tracing 1 lens unit before first surface
        z_start = -1
        wavelength = self.optic.primary_wavelength
        y, u = self.tracer.trace_generic(1.0, 0.0, z_start, wavelength,
                                         reverse=True)
        F1 = y[-1] / u[-1]
        return F1[0]

    def F2(self):
        """Calculate the back focal point location

        Returns:
            float: back focal point location
        """
        # start tracing 1 lens unit before first surface
        z_start = self.surfaces.positions[1] - 1
        wavelength = self.optic.primary_wavelength
        y, u = self.tracer.trace_generic(1.0, 0.0, z_start[0], wavelength)
        F2 = -y[-1] / u[-1]
        return F2[0]

    def P1(self):
        """Calculate the front principle plane location

        Returns:
            float: front principle plane location
        """
        return self.F1() - self.f1()

    def P2(self):
        """Calculate the back principle plane location

        Returns:
            float: back principle plane location
        """
        return self.F2() - self.f2()

    def N1(self):
        """Calculate the front nodal plane location

        Returns:
            float: front nodal plane location
        """
        return self.P1() + self.f1() + self.f2()

    def N2(self):
        """Calculate the back nodal plane location

        Returns:
            float: back nodal plane location
        """
        return self.P2() + self.f1() + self.f2()

    def EPL(self):
        """Calculate the entrance pupil location in global coordinates

        Returns:
            float: entrance pupil position relative to first surface, which
                lies at z=0 by definition.
        """
        stop_index = self.surfaces.stop_index
        if stop_index == 0:
            return self.surfaces.positions[1]

        y0 = 0
        u0 = 0.1
        pos = self.surfaces.positions
        z0 = pos[-1] - pos[stop_index]
        wavelength = self.optic.primary_wavelength

        # trace from center of stop on axis
        y, u = self.tracer.trace_generic(y0, u0, z0, wavelength, reverse=True,
                                         skip=stop_index)

        loc_relative = y[-1] / u[-1]
        try:
            return loc_relative[0]
        except IndexError:
            return loc_relative

    def EPD(self):
        """Caculate the entrance pupil diameter

        Returns:
            float: entrance pupil diameter
        """
        ap_type = self.optic.aperture.ap_type
        ap_value = self.optic.aperture.value

        if ap_type == 'EPD':
            return ap_value

        elif ap_type == 'imageFNO':
            return self.f2() / ap_value

        elif ap_type == 'objectNA':
            obj_z = self.optic.object_surface.geometry.cs.z
            wavelength = self.optic.primary_wavelength
            n0 = self.optic.object_surface.material_post.n(wavelength)
            u0 = be.arcsin(ap_value / n0)
            z = self.EPL() - obj_z
            return 2 * z * be.tan(u0)

    def XPL(self):
        """Calculate the exit pupil location

        Returns:
            float: exit pupil location relative to the image surface
        """
        stop_index = self.surfaces.stop_index
        num_surfaces = len(self.surfaces.surfaces)
        if stop_index == num_surfaces-2:
            positions = self.optic.surface_group.positions
            loc_relative = positions[-2] - positions[-1]
            return loc_relative[0]

        z_start = self.surfaces.positions[stop_index]
        wavelength = self.optic.primary_wavelength
        y, u = self.tracer.trace_generic(0.0, 0.1, z_start[0], wavelength,
                                         skip=stop_index+1)

        loc_relative = -y[-1] / u[-1]
        return loc_relative[0]

    def XPD(self):
        """Calculate the exit pupil diameter

        Returns:
            float: exit pupil diameter
        """
        # find marginal ray height at image surface
        ya, ua = self.marginal_ray()
        yi = ya[-1]
        ui = ua[-1]

        # find distance from image surface to exit pupil location
        xpl = self.XPL()

        # propagate marginal ray to this location
        yxp = yi + ui * xpl
        return 2 * yxp[0]

    def FNO(self):
        """Calculate the image-space F-number

        Returns:
            float: image-space F-number
        """
        ap_type = self.optic.aperture.ap_type
        if ap_type == 'imageFNO':
            return self.optic.aperture.value
        else:
            return self.f2() / self.EPD()

    def magnification(self):
        '''Calculate the magnification

        Returns:
            float: the system magnification
        '''
        _, ua = self.marginal_ray()
        n = self.optic.n()
        mag = n[0] * ua[0] / (n[-1] * ua[-1])
        return mag[0]

    def invariant(self):
        """Calculate the Lagrange invariant

        Returns:
            float: the Lagrange invariant
        """
        ya, ua = self.marginal_ray()
        yb, ub = self.chief_ray()
        n = self.optic.n()
        inv = yb[1] * n[1] * ua[1] - ya[1] * n[1] * ub[1]
        return inv[0]

    def marginal_ray(self):
        """Find the marginal ray heights and angles

        Returns:
            tuple: marginal ray heights and angles as type np.ndarray
        """
        EPD = self.EPD()
        obj_z = self.surfaces.positions[1] - 10  # 10 mm before first surface
        if self.optic.object_surface.is_infinite:
            ya = EPD / 2
            ua = 0
        else:
            obj_z = self.optic.object_surface.geometry.cs.z
            z = self.EPL() - obj_z
            ya = 0
            ua = EPD / (2 * z)

        wavelength = self.optic.primary_wavelength
        return self.tracer.trace_generic(ya, ua, obj_z[0], wavelength)

    def chief_ray(self):
        """Find the chief ray heights and angles

        Returns:
            tuple: chief ray heights and angles as type np.ndarray
        """
        stop_index = self.optic.surface_group.stop_index
        y0 = 0
        u0 = 0.1
        pos = self.optic.surface_group.positions
        z0 = pos[-1] - pos[stop_index]
        wavelength = self.optic.primary_wavelength

        # trace from center of stop on axis
        y, u = self.tracer.trace_generic(y0, u0, z0, wavelength, reverse=True,
                                         skip=stop_index)

        max_field = self.optic.fields.max_y_field

        if self.optic.field_type == 'object_height':
            u1 = 0.1 * max_field / y[-1]
        elif self.optic.field_type == 'angle':
            u1 = 0.1 * be.tan(be.deg2rad(max_field)) / u[-1]

        yn, un = self.tracer.trace_generic(y0, u1, z0, wavelength,
                                           reverse=True, skip=stop_index)

        # trace in forward direction
        z0 = self.optic.surface_group.positions[1]

        return self.tracer.trace_generic(-yn[-1, 0], un[-1, 0], z0[0],
                                         wavelength)
