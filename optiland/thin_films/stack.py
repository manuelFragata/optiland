import numpy as np


class ThinFilmStack:
    def __init__(self, incident_material, substrate_material):
        self._inc_mat = incident_material
        self._sub_mat = substrate_material
        self._layers = []

        self._reflectance_cache = {}
        self._jones_matrix_cache = {}
        self._char_mat_cache = {}

    @property
    def inc_mat(self):
        return self._inc_mat

    @inc_mat.setter
    def inc_mat(self, material):
        self._inc_mat = material
        self._clear_cache()

    @property
    def sub_mat(self):
        return self._sub_mat

    @sub_mat.setter
    def sub_mat(self, material):
        self._sub_mat = material
        self._clear_cache()

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, layers):
        self._layers = layers
        self._clear_cache()

    def grow(self, layer):
        """Grow a layer on the stack."""
        self.layers.append(layer)

    def reflectance(self, wavelength, aoi, pol):
        """Calculate the reflectance of the stack."""
        cache_key = (wavelength, aoi, pol)
        if cache_key in self._reflectance_cache:
            return self._reflectance_cache[cache_key]

        y0 = self._admittance(self.inc_mat.n(wavelength), aoi, pol)
        ys = self._admittance(self.sub_mat.n(wavelength), aoi, pol)
        mat = self._characteristic_matrix(wavelength, aoi, pol)

        B = mat[0, 0] + mat[0, 1] * ys
        C = mat[1, 0] + mat[1, 1] * ys

        r = (y0 * B - C) / (y0 * B + C)
        self._reflectance_cache[cache_key] = r
        return r

    def jones_matrix(self, wavelength, aoi):
        """Calculate the Jones matrix of the stack."""
        pass

    def _characteristic_matrix(self, wavelength, aoi, pol):
        thetas = self._compute_thetas(wavelength, aoi)
        m = np.eye(2)
        for k, layer in enumerate(self.layers):
            m_layer = layer.characteristic_matrix(wavelength, thetas[k], pol)
            m = np.dot(m, m_layer)
        return m

    def _compute_thetas(self, wavelength, aoi):
        """Compute angles through all layers."""
        n0 = self.inc_mat.n(wavelength)
        n = np.array([layer.material.n(wavelength) for layer in self.layers])
        return np.arcsin(n0 / n * np.sin(aoi))

    def _admittance(self, n, aoi, pol):
        """Compute admittance of incident and substrate materials."""
        cos = np.cos(aoi)
        if pol == 's':
            return n * cos
        elif pol == 'p':
            return n / cos

    def _clear_cache(self):
        self._reflectance_cache = {}
        self._jones_matrix_cache = {}
        self._char_mat_cache = {}
