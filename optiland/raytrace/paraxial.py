import optiland.backend as be


class ParaxialRayTracer:
    """Paraxial Ray Tracer

    This class performs paraxial ray tracing through an optical system.

    Args:
        surface_group (SurfaceGroup): The surface group to trace rays through.
    """
    def __init__(self, optic):
        self.optic = optic

    def trace(self, Hy, Py, wavelength):
        """
        Trace paraxial ray through the optical system based on specified field
        and pupil coordinates.

        Args:
            Hy (float): Normalized field coordinate.
            Py (float): Normalized pupil coordinate.
            wavelength (float): Wavelength of the light.
        """
        Hy = self._process_input(Hy)
        Py = self._process_input(Py)
        wavelength = self._process_input(wavelength)

        EPL = self.optic.paraxial.EPL()
        EPD = self.optic.paraxial.EPD()

        y1 = Py * EPD / 2

        y0, z0 = self._get_object_position(Hy, y1, EPL)
        u0 = (y1 - y0) / (EPL - z0)

        self.trace_generic(y0, u0, z0, wavelength)

    def trace_generic(self, y, u, z, wavelength, reverse=False, skip=0):
        """
        Trace generically-defined paraxial rays through the optical system.

        Args:
            y (float or array-like): The initial height(s) of the rays.
            u (float or array-like): The initial slope(s) of the rays.
            z (float or array-like): The initial axial position(s) of the rays.
            wavelength (float): The wavelength of the rays.
            reverse (bool, optional): If True, trace the rays in reverse
                direction. Defaults to False.
            skip (int, optional): The number of surfaces to skip during
                tracing. Defaults to 0.

        Returns:
            tuple: A tuple containing the final height(s) and slope(s) of the
                rays after tracing.
        """
        # TODO: this is a workaround to maintain performance while using
        # a configurable backend. Move this to a dedicated tracer class.
        self._process_input(y)
        self._process_input(u)
        self._process_input(z)

        R = self.optic.surface_group.radii  # radii of curvature
        n = self.optic.n(wavelength)  # refractive index at wavelength
        position = be.ravel(self.optic.surface_group.positions)  # z positions
        surfaces = self.optic.surface_group.surfaces  # surface types
        num_surfaces = len(surfaces)

        if reverse:
            R = -R[::-1]
            n = be.concatenate((n[::-1][1:], [n[0]]))
            position = be.abs(position[::-1] - position[::-1][0])
            surfaces = surfaces[::-1]

        t = be.abs(be.diff(position))
        t[be.isinf(t)] = 0.0

        power = (n[1:] - n[:-1]) / R[1:]

        y_out = [be.array([y])]
        u_out = [be.array([u])]

        for i in range(skip, num_surfaces):
            if be.isinf(position[i]):
                continue

            shift = position[i] - z
            z = position[i]

            if reverse:
                y = y + t[i] * u
            else:
                y = y + shift * u

            if surfaces[i].is_reflective:
                u = -u - 2 * y / R[i]
            else:
                n1, n2 = (n[i], n[i + 1]) if reverse else (n[i - 1], n[i])
                power_index = i if reverse else i - 1
                u = (n1 * u - y * power[power_index]) / n2

            y_out.append(be.array([y]))
            u_out.append(be.array([u]))

        return be.array(y_out), be.array(u_out)

    def _get_object_position(self, Hy, y1, EPL):
        """
        Calculate the position of the object in the paraxial optical system.

        Args:
            Hy (float): The normalized field height.
            y1 (ndarray): The initial y-coordinate of the ray.
            EPL (float): The effective focal length of the lens.

        Returns:
            tuple: A tuple containing the y and z coordinates of the object
                position.

        Raises:
            ValueError: If the field type is "object_height" and the object is
                at infinity.
        """
        obj = self.optic.object_surface
        field_y = self.optic.fields.max_field * Hy

        if obj.is_infinite:
            if self.optic.field_type == 'object_height':
                raise ValueError('Field type cannot be "object_height" for an '
                                 'object at infinity.')

            y = -be.tan(be.deg2rad(field_y)) * EPL
            z = self.optic.surface_group.positions[1]

            y0 = y1 + y
            z0 = be.ones_like(y1) * z
        else:
            if self.optic.field_type == 'object_height':
                y = -field_y
                z = obj.geometry.cs.z

                y0 = be.ones_like(y1) * y
                z0 = be.ones_like(y1) * z

            elif self.optic.field_type == 'angle':
                y = -be.tan(be.deg2rad(field_y))
                z = self.optic.surface_group.positions[0]

                y0 = y1 + y
                z0 = be.ones_like(y1) * z

        return y0, z0

    def _process_input(self, x):
        """
        Process input to ensure it is a numpy array.

        Args:
            x (float or array-like): The input to process.

        Returns:
            np.ndarray: The processed input.
        """
        if isinstance(x, (int, float)):
            return be.array([x])
        else:
            return be.array(x)
