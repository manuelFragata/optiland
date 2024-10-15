import numpy as np


class ThinFilmLayer:
    def __init__(self, thickness, material):
        self._thickness = thickness
        self._material = material

        self._char_mat_cache = {}

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        self._thickness = thickness
        self._clear_cache()

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, material):
        self._material = material
        self._clear_cache()

    def characteristic_matrix(self, wavelength, theta, pol):
        """Calculate the characteristic matrix of the layer."""
        cache_key = (wavelength, theta, pol)

        # Check if the result is already cached
        if cache_key in self._char_mat_cache:
            return self._char_mat_cache[cache_key]

        # Calculate the characteristic matrix
        n = self.material.n(wavelength) + 1j * self.material.k(wavelength)
        admittance = self._admittance(pol, theta, n)
        phase = self._phase_thickness(wavelength, theta, n)
        mat = np.array([[np.cos(phase), 1j / admittance * np.sin(phase)],
                        [1j * admittance * np.sin(phase), np.cos(phase)]])

        # Cache the result
        self._char_mat_cache[cache_key] = mat

        return mat

    def _phase_thickness(self, wavelength, theta, n):
        """Calculate the phase thickness of the layer."""
        return 2 * np.pi * n * self.thickness * np.cos(theta) / wavelength

    def _admittance(self, pol, theta, n):
        """Calculate the characteristic admittance of the layer."""
        if pol == 's':
            return n * np.cos(theta)
        elif pol == 'p':
            return n / np.cos(theta)

    def _clear_cache(self):
        self._char_mat_cache = {}
