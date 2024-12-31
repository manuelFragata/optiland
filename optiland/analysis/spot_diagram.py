"""Spot Diagram Analysis

This module provides a spot diagram analysis for optical systems.

Kramer Harrison, 2024
"""
import matplotlib.pyplot as plt
import optiland.backend as be


class SpotDiagram:
    """Spot diagram class

    This class generates data and plots real ray intersection locations
    on the final optical surface in an optical system. These plots
    are purely geometric and give an indication of the blur produced
    by aberrations in the system.

    Attributes:
        optic (optic.Optic): instance of the optic object to be assessed
        fields (tuple): fields at which data is generated
        wavelengths (tuple[float]): wavelengths at which data is generated
        num_rings (int): number of rings in pupil distribution for ray tracing
        data (List): contains spot data in a nested list. Data is ordered as
            field (dim 0), wavelength (dim 1), then x, y and intensity data
            (dim 2).
    """

    def __init__(self, optic, fields='all', wavelengths='all', num_rings=6,
                 distribution='hexapolar'):
        """Create an instance of SpotDiagram

        Note:
            The constructor also generates all data that may later be used for
            plotting

        Args:
            optic (optic.Optic): instance of the optic object to be assessed
            fields (tuple or str): fields at which data is generated.
                If 'all' is passed, then all field points are considered.
                Default is 'all'.
            wavelengths (str or tuple[float]): wavelengths at which data is
                generated. If 'all' is passed, then all wavelengths are
                considered. Default is 'all'.
            num_rings (int): number of rings in pupil distribution for ray
                tracing. Default is 6.
            distribution (str): pupil distribution type for ray tracing.
                Default is 'hexapolar'.

        Returns:
            None
        """
        self.optic = optic
        self.fields = fields
        self.wavelengths = wavelengths
        if self.fields == 'all':
            self.fields = self.optic.fields.get_field_coords()

        if self.wavelengths == 'all':
            self.wavelengths = self.optic.wavelengths.get_wavelengths()

        result = self._generate_data(
            self.fields,
            self.wavelengths,
            num_rays=num_rings,
            distribution=distribution
        )

        self.data = result[0]
        self.centroids = result[1]
        self.geometric_spot_radius = result[2]
        self.rms_spot_radius = result[3]

    def view(self, figsize=(12, 4)):
        """View the spot diagram

        Args:
            figsize (tuple): the figure size of the output window.
                Default is (12, 4).

        Returns:
            None
        """
        N = len(self.fields)
        num_rows = (N + 2) // 3

        fig, axs = plt.subplots(num_rows, 3,
                                figsize=(figsize[0], num_rows*figsize[1]),
                                sharex=True, sharey=True)
        axs = axs.flatten()

        # subtract centroid and find limits
        data = self._prepare_data(self.data)
        geometric_size = self.geometric_spot_radius
        axis_lim = be.max(be.array(geometric_size))
        axis_lim = be.to_numpy(axis_lim)

        # plot wavelengths for each field
        for k, field_data in enumerate(data):
            self._plot_field(axs[k], field_data, self.fields[k],
                             axis_lim, self.wavelengths)

        # remove empty axes
        for k in range(N, num_rows * 3):
            fig.delaxes(axs[k])

        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')

        plt.tight_layout()
        plt.show()

    def geometric_spot_radius(self):
        """Geometric spot radius of each spot

        Returns:
            geometric_size (List): Geometric spot radius for field and
                wavelength
        """
        return self.geometric_spot_radius

    def rms_spot_radius(self):
        """Root mean square (RMS) spot radius of each spot

        Returns:
            rms (List): RMS spot radius for each field and wavelength.
        """
        return self.rms_spot_radius

    @staticmethod
    def _geometric_spot_radius(data, centroids):
        """Geometric spot radius of each spot

        Returns:
            geometric_size (List): Geometric spot radius for field and
                wavelength
        """
        geometric_size = []
        for k, field_data in enumerate(data):
            geometric_size_field = []
            for wave_data in field_data:
                x = wave_data[0] - centroids[k][0]
                y = wave_data[1] - centroids[k][1]
                r = be.sqrt(x**2 + y**2)
                geometric_size_field.append(be.max(r))
            geometric_size.append(geometric_size_field)
        return geometric_size

    @staticmethod
    def _rms_spot_radius(data, centroids):
        """Root mean square (RMS) spot radius of each spot

        Returns:
            rms (List): RMS spot radius for each field and wavelength.
        """
        rms = []
        for k, field_data in enumerate(data):
            rms_field = []
            for wave_data in field_data:
                x = wave_data[0] - centroids[k][0]
                y = wave_data[1] - centroids[k][1]
                r2 = x**2 + y**2
                rms_field.append(be.sqrt(be.mean(r2)))
            rms.append(rms_field)
        return rms

    def _compute_centroids(self, data):
        norm_index = self.optic.wavelengths.primary_index
        centroids = []
        for field_data in data:
            centroid_x = be.mean(field_data[norm_index][0])
            centroid_y = be.mean(field_data[norm_index][1])
            centroids.append((centroid_x, centroid_y))
        return centroids

    def _generate_data(self, fields, wavelengths, num_rays=100,
                       distribution='hexapolar'):
        """
        Generate spot data for the given fields and wavelengths.

        Args:
            fields (List): A list of fields.
            wavelengths (List): A list of wavelengths.
            num_rays (int, optional): The number of rays to generate.
                Defaults to 100.
            distribution (str, optional): The distribution type.
                Defaults to 'hexapolar'.

        Returns:
            data (List): A nested list of spot intersection data for each
                field and wavelength.
        """
        data = []
        for field in fields:
            field_data = []
            for wavelength in wavelengths:
                field_data.append(
                    self._generate_field_data(
                        field, wavelength, num_rays, distribution
                    )
                )
            data.append(field_data)

        centroids = self._compute_centroids(data)
        geo_spot_size = self._geometric_spot_radius(data, centroids)
        rms_spot_size = self._rms_spot_radius(data, centroids)

        return data, centroids, geo_spot_size, rms_spot_size

    def _generate_field_data(self, field, wavelength, num_rays=100,
                             distribution='hexapolar'):
        """
        Generates spot data for a given field and wavelength.

        Args:
            field (tuple): Tuple containing the field coordinates in (x, y).
            wavelength (float): The wavelength of the field.
            num_rays (int, optional): The number of rays to generate.
                Defaults to 100.
            distribution (str, optional): The distribution pattern of the
                rays. Defaults to 'hexapolar'.

        Returns:
            list: A list containing the x, y, and intensity values of the
                generated spot data.
        """
        self.optic.trace(*field, wavelength, num_rays, distribution)
        x = self.optic.surface_group.x[-1, :]
        y = self.optic.surface_group.y[-1, :]
        intensity = self.optic.surface_group.intensity[-1, :]
        return [x, y, intensity]

    def _plot_field(self, ax, field_data, field, axis_lim,
                    wavelengths, buffer=1.05):
        """
        Plot the field data on the given axis.

        Parameters:
            ax (matplotlib.axes.Axes): The axis to plot the field data on.
            field_data (list): List of tuples containing x, y, and intensity
                data points.
            field (tuple): Tuple containing the Hx and Hy field values.
            axis_lim (float): Limit of the x and y axis.
            wavelengths (list): List of wavelengths corresponding to the
                field data.
            buffer (float, optional): Buffer factor to extend the axis limits.
                Default is 1.05.

        Returns:
            None
        """
        markers = ['o', 's', '^']
        for k, points in enumerate(field_data):
            x, y, intensity = points
            mask = intensity != 0
            ax.scatter(x[mask], y[mask], s=10,
                       label=f'{wavelengths[k]:.4f} µm',
                       marker=markers[k % 3], alpha=0.7)
            ax.axis('square')
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_xlim((-axis_lim*buffer, axis_lim*buffer))
            ax.set_ylim((-axis_lim*buffer, axis_lim*buffer))
        ax.set_title(f'Hx: {field[0]:.3f}, Hy: {field[1]:.3f}')

    def _prepare_data(self, data):
        """Prepare the data for visualization."""
        new_data = []
        for k, field_data in enumerate(data):
            subdata = []
            for wave_data in field_data:
                x = be.to_numpy(wave_data[0] - self.centroids[k][0])
                y = be.to_numpy(wave_data[1] - self.centroids[k][1])
                intensity = be.to_numpy(wave_data[2])
                subdata.append([x, y, intensity])
            new_data.append(subdata)
        return new_data
