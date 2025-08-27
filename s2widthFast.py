import numpy as np
import math
import matplotlib.pyplot as plt
from numba import njit, prange


# -------------------------------------------------------
# Numba-compatible helpers
# -------------------------------------------------------

@njit(fastmath=True)
def integrand_numba(t, dt, sigma, k1, k2):
    """Compute integrand for given parameters at scalar t."""
    erf1 = math.erf((t - dt / 2.0) / (math.sqrt(2.0) * sigma))
    erf2 = math.erf((t + dt / 2.0) / (math.sqrt(2.0) * sigma))
    return (erf1 - erf2) * math.cos((k1 - k2) * t)


@njit(fastmath=True)
def Rho_p_numba(dt, n, sigma, p, tmin=-5.0, tmax=5.0, Npts=500):
    """Compute Rho_p using trapezoidal integration."""
    k1 = 2.0 * math.pi * p / dt
    k2 = 2.0 * math.pi * (n - p) / dt
    t_values = np.linspace(tmin, tmax, Npts)
    integrand_values = np.empty(Npts, dtype=np.float64)

    for i in range(Npts):
        integrand_values[i] = integrand_numba(t_values[i], dt, sigma, k1, k2)

    # trapezoidal integration
    integral = 0.0
    for i in range(Npts - 1):
        integral += 0.5 * (integrand_values[i] + integrand_values[i + 1]) * (t_values[i + 1] - t_values[i])

    return integral


@njit(parallel=True, fastmath=True)
def pdf_1d_numba(n, sigma, p, dt_values):
    """Return PDF for array of dt_values at fixed n, sigma, p."""
    pdf = np.empty(len(dt_values), dtype=np.float64)
    for i in prange(len(dt_values)):
        pdf[i] = Rho_p_numba(dt_values[i], n, sigma, p)
    return pdf


# -------------------------------------------------------
# Width Model Class
# -------------------------------------------------------

class WidthModel:
    def __init__(self, M=0.01, p=1, C=0.0, dC=0.0):
        """
        Parameters:
            M : diffusion coefficient (controls sigma ~ sqrt(t))
            p : integer index for PDF calculation
            C : constant offset for widths
            dC: optional Gaussian smearing width
        """
        self.M = M
        self.p = p
        self.C = C
        self.dC = dC

    def shift_pdf(self, widths, pdf_values, shift):
        """Shift PDF in width space by a constant offset."""
        return widths + shift, pdf_values

    def smear_pdf(self, widths, pdf_values, smear_sigma):
        """Apply Gaussian smearing to PDF."""
        from scipy.ndimage import gaussian_filter1d
        return widths, gaussian_filter1d(pdf_values, smear_sigma)

    def normalize_pdf(self, pdf_values):
        """Normalize PDF to unit integral."""
        total = np.trapz(pdf_values)
        if total > 0:
            return pdf_values / total
        return pdf_values

    def pdf_1d(self, n, drift_time, widths):
        """Compute 1D PDF given n, drift_time, widths."""
        sigma = self.M * np.sqrt(drift_time)
        pdf_values = pdf_1d_numba(n, sigma, self.p, widths)
        widths_shifted, pdf_values = self.shift_pdf(widths, pdf_values, self.C)
        if self.dC > 0:
            widths_shifted, pdf_values = self.smear_pdf(widths_shifted, pdf_values, self.dC)
        return widths_shifted, self.normalize_pdf(pdf_values)

    def plot_pdf_1d(self, n, drift_time, widths, show=True):
        """Plot 1D PDF."""
        widths_shifted, pdf_values = self.pdf_1d(n, drift_time, widths)
        plt.figure()
        plt.plot(widths_shifted, pdf_values, label=f"n={n}, drift={drift_time}")
        plt.xlabel("Width")
        plt.ylabel("PDF")
        plt.legend()
        if show:
            plt.show()
        return widths_shifted, pdf_values
