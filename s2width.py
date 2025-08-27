# s2width.py

import numpy as np
import scipy.special as sp
import scipy.integrate as integrate
import math
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class WidthModel:
    """
    Model for S2 width distributions from longitudinal diffusion.
    """

    def __init__(self, Dl, vd, C=0.0, dC=0.0, p=0.5):
        """
        Initialize the WidthModel with physical constants.

        Parameters:
        - Dl: longitudinal diffusion constant
        - vd: drift velocity
        - C: constant offset for PDF shifting (width' = sqrt(width^2 + C^2))
        - dC: Gaussian smoothing scale in the same units as width
        - p: symmetry parameter (default 0.5)
        """
        self.Dl = Dl
        self.vd = vd
        self.C = C
        self.dC = dC
        self.p = p
        # M = 2 * sqrt(Dl) / vd
        self.M = 2.0 * np.sqrt(Dl) / vd

    # ------------------------
    # Internal helpers
    # ------------------------
    @staticmethod
    def _integrand(t, dt, sigma, k1, k2):
        erf1 = sp.erf((t - dt / 2) / (np.sqrt(2) * sigma))
        erf2 = sp.erf((t + dt / 2) / (np.sqrt(2) * sigma))
        erfc1 = sp.erfc((t + dt / 2) / (np.sqrt(2) * sigma))
        term1 = (1 + erf1) * erfc1
        term2 = erf2 - erf1
        return np.exp(-t**2 / sigma**2) * term1**k1 * term2**k2

    @staticmethod
    def _Rho_p(dt, n, sigma, p):
        """
        Core PDF piece from the diffusion derivation (scalar in dt).
        """
        k1 = (1 - p) / 2 * n
        k2 = p * n - 2
        if k2 < 0 or k1 < 0:
            return 0.0

        coeff = math.factorial(n) / (math.factorial(int(k1))**2 * math.factorial(int(k2)))
        exp_term = np.exp(-dt**2 / (4 * sigma**2))
        prefactor = 1 / (2 * np.pi * sigma**2 * 2**(n - 2))

        integral, _ = integrate.quad(
            WidthModel._integrand,
            -5 * sigma,
            5 * sigma,
            epsrel=1e-5,
            args=(dt, sigma, k1, k2),
        )
        return coeff * exp_term * prefactor * integral

    @staticmethod
    def _shift_pdf(X, Y, C):
        """
        Transform PDF under X' = sqrt(X^2 + C^2), with Jacobian correction,
        and normalize the transformed PDF.
        """
        X_new = np.linspace(0, np.sqrt(max(X)**2 + C**2), len(X))
        X_original = np.sqrt(np.maximum(X_new**2 - C**2, 0))
        interp_func = interp1d(X, Y, kind='linear', fill_value=0, bounds_error=False)
        Y_interpolated = interp_func(X_original)
        jacobian = np.zeros_like(X_new)
        nonzero = X_new > 0
        jacobian[nonzero] = X_original[nonzero] / X_new[nonzero]
        Y_new = Y_interpolated * jacobian

        area = np.trapz(Y_new, X_new)
        if area > 0:
            Y_new /= area
        return X_new, Y_new

    def _pdf_1d(self, n, drift_time, num_points=1000):
        """
        Compute the 1D PDF (width) for a given n and drift_time.
        Returns (widths, pdf) already shifted by C and optionally smoothed by dC.
        """
        sigma = self.M * np.sqrt(drift_time)
        lim = 10 * sigma
        if lim <= 0:
            # Degenerate case
            return np.array([0.0, 1.0]), np.array([1.0, 0.0])

        dt_values = np.linspace(0, lim, num_points)
        distribution = np.array([self._Rho_p(dt, n, sigma, self.p) for dt in dt_values])

        # Shift by C with Jacobian and renormalize
        dt_values, distribution = self._shift_pdf(dt_values, distribution, self.C)

        # Optional smoothing (convert physical dC into pixel sigma)
        if self.dC > 0 and len(dt_values) > 1:
            pixel_scale = dt_values[1] - dt_values[0]
            smoothed_sigma = self.dC / pixel_scale
            if smoothed_sigma > 0:
                distribution = gaussian_filter1d(distribution, smoothed_sigma, mode='constant', cval=0)

        # Final normalization
        area = np.trapz(distribution, dt_values)
        if area > 0:
            distribution /= area

        return dt_values, distribution

    # ------------------------
    # Public plotting / building methods
    # ------------------------
    def plot_pdf_1d(self, n, drift_time, widths=None, num_points=1000, show=True):
        """
        Plot 1D width PDF for a given number of electrons (n) and drift_time.

        - If `widths` is None, computes the PDF on an internal adaptive grid (num_points).
        - If `widths` is provided (1D array), interpolates the PDF onto that grid for plotting.
        """
        base_x, base_pdf = self._pdf_1d(n, drift_time, num_points=num_points)

        if widths is not None:
            y = np.interp(widths, base_x, base_pdf, left=0.0, right=0.0)
            x = widths
        else:
            x, y = base_x, base_pdf

        if show:
            plt.figure(figsize=(7, 4))
            plt.plot(x, y, label=f"N={n}, t={drift_time}")
            plt.xlabel("Width")
            plt.ylabel("Probability density")
            plt.title("1D Width PDF")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        return x, y

    def plot_pdf_2d_fixedN(self, n, drift_times, widths):
        """
        Plot the 2D PDF slice for a fixed N across drift times and widths.
        """
        pdf = np.zeros((len(drift_times), len(widths)))
        for j, dt in enumerate(drift_times):
            base_x, dist = self._pdf_1d(n, dt, num_points=len(widths))
            pdf[j, :] = np.interp(widths, base_x, dist, left=0, right=0)

        plt.figure(figsize=(8, 6))
        plt.imshow(
            pdf,
            aspect="auto",
            origin="lower",
            extent=[widths[0], widths[-1], drift_times[0], drift_times[-1]],
            cmap="viridis",
        )
        plt.colorbar(label="Probability density")
        plt.xlabel("Width")
        plt.ylabel("Drift time")
        plt.title(f"2D Width PDF at N={n}")
        plt.show()

    def build_pdf_3d(self, n_values, drift_times, widths):
        """
        Construct a 3D PDF P(width | drift_time, N) for varying number of electrons.

        Returns:
        - pdf: ndarray of shape (len(n_values), len(drift_times), len(widths))
        """
        pdf = np.zeros((len(n_values), len(drift_times), len(widths)))

        for i, n in enumerate(n_values):
            for j, dt in enumerate(drift_times):
                base_x, dist = self._pdf_1d(n, dt, num_points=len(widths))
                pdf[i, j, :] = np.interp(widths, base_x, dist, left=0, right=0)

        # Normalize each (n, drift_time) slice
        for i in range(len(n_values)):
            for j in range(len(drift_times)):
                area = np.trapz(pdf[i, j, :], widths)
                if area > 0:
                    pdf[i, j, :] /= area

        return pdf
