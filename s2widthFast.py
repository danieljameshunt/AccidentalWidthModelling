# s2width.py
import math
import time
import numpy as np
import scipy.special as sp
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d, RegularGridInterpolator
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from typing import Sequence, Tuple
from tqdm.auto import tqdm
from numba import njit, prange

# ------------------------
# Numba-compatible helpers
# ------------------------

@njit(cache=True)
def factorial_numba(n: int) -> float:
    if n <= 1:
        return 1.0
    f = 1.0
    for i in range(2, n + 1):
        f *= i
    return f

@njit(cache=True, fastmath=True)
def integrand_numba(t: float, dt: float, sigma: float, k1: float, k2: float) -> float:
    arg1 = (t - dt / 2.0) / (math.sqrt(2.0) * sigma)
    arg2 = (t + dt / 2.0) / (math.sqrt(2.0) * sigma)
    erf1 = math.erf(arg1)
    erf2 = math.erf(arg2)
    erfc1 = 1.0 - erf2
    term1 = (1.0 + erf1) * erfc1
    term2 = erf2 - erf1
    t1pow = term1 ** k1 if term1 > 0 else 0.0
    t2pow = term2 ** k2 if term2 > 0 else 0.0
    return math.exp(- (t * t) / (sigma * sigma)) * t1pow * t2pow

@njit(cache=True, fastmath=True)
def Rho_p_trapz_numba(dt: float, n: int, sigma: float, p: float,
                      tmin: float = -5.0, tmax: float = 5.0, Npts: int = 800) -> float:
    if dt <= 0.0:
        return 0.0
    k1 = (1.0 - p) / 2.0 * n
    k2 = p * n - 2.0
    if k1 < 0.0 or k2 < 0.0:
        return 0.0
    ik1 = int(k1)
    ik2 = int(k2)
    coeff = factorial_numba(n) / (factorial_numba(ik1) ** 2 * factorial_numba(ik2))
    exp_term = math.exp(- (dt * dt) / (4.0 * sigma * sigma))
    prefactor = 1.0 / (2.0 * math.pi * sigma * sigma * math.pow(2.0, (n - 2)))
    step = (tmax - tmin) / (Npts - 1)
    t = tmin
    s = 0.5 * integrand_numba(t, dt, sigma, k1, k2)
    for i in range(1, Npts - 1):
        t = tmin + i * step
        s += integrand_numba(t, dt, sigma, k1, k2)
    t = tmax
    s += 0.5 * integrand_numba(t, dt, sigma, k1, k2)
    integral = s * step
    return coeff * exp_term * prefactor * integral

@njit(cache=True, parallel=True, fastmath=True)
def pdf_1d_numba(n: int, sigma: float, p: float, widths: np.ndarray, Npts: int = 800) -> np.ndarray:
    out = np.zeros_like(widths)
    for i in prange(widths.shape[0]):
        out[i] = Rho_p_trapz_numba(widths[i], n, sigma, p, -5.0, 5.0, Npts)
    return out

# ------------------------
# WidthModel
# ------------------------

class WidthModel:
    def __init__(self, Dl: float, vd: float, C: float = 0.0, dC: float = 0.0, p: float = 0.5, numba_Npts: int = 800):
        self.Dl = Dl
        self.vd = vd
        self.C = C
        self.dC = dC
        self.p = p
        self.M = np.sqrt(2*Dl) / vd
        self.numba_Npts = int(numba_Npts)

    @staticmethod
    def _integrand(t: float, dt: float, sigma: float, k1: float, k2: float) -> float:
        erf1 = sp.erf((t - dt / 2) / (np.sqrt(2) * sigma))
        erf2 = sp.erf((t + dt / 2) / (np.sqrt(2) * sigma))
        erfc1 = sp.erfc((t + dt / 2) / (np.sqrt(2) * sigma))
        term1 = (1.0 + erf1) * erfc1
        term2 = erf2 - erf1
        return np.exp(-t**2 / sigma**2) * term1**k1 * term2**k2

    @staticmethod
    def _Rho_p(dt: float, n: int, sigma: float, p: float) -> float:
        k1 = (1 - p) / 2 * n
        k2 = p * n - 2
        if k2 < 0 or k1 < 0:
            return 0.0
        coeff = math.factorial(n) / (math.factorial(int(k1))**2 * math.factorial(int(k2)))
        exp_term = np.exp(-dt**2 / (4 * sigma**2))
        prefactor = 1 / (2 * np.pi * sigma**2 * 2**(n - 2))
        integral, _ = integrate.quad(WidthModel._integrand, -5 * sigma, 5 * sigma, epsrel=1e-5,
                                     args=(dt, sigma, k1, k2))
        return coeff * exp_term * prefactor * integral

    @staticmethod
    def _shift_pdf(X: np.ndarray, Y: np.ndarray, C: float) -> Tuple[np.ndarray, np.ndarray]:
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

    def _pdf_1d(self, n: int, drift_time: float, num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        sigma = self.M * np.sqrt(drift_time)
        lim = 10 * sigma
        if lim <= 0:
            return np.array([0.0]), np.array([1.0])
        dt_values = np.linspace(0, lim, num_points)
        distribution = np.array([self._Rho_p(dt, n, sigma, self.p) for dt in dt_values])
        dt_values, distribution = self._shift_pdf(dt_values, distribution, self.C)
        if self.dC > 0 and len(dt_values) > 1:
            pixel_scale = dt_values[1] - dt_values[0]
            smoothed_sigma = self.dC / pixel_scale
            if smoothed_sigma > 0:
                distribution = gaussian_filter1d(distribution, smoothed_sigma, mode='constant', cval=0)
        area = np.trapz(distribution, dt_values)
        if area > 0:
            distribution /= area
        return dt_values, distribution

    def pdf_1d_numba_wrapper(self, n: int, drift_time: float, widths: np.ndarray = None, num_points: int = None) -> Tuple[np.ndarray, np.ndarray]:
        sigma = self.M * np.sqrt(drift_time)
        if widths is None:
            if num_points is None:
                num_points = 1000
            lim = 10.0 * sigma
            widths = np.linspace(0.0, lim, int(num_points)).astype(float)
        else:
            widths = np.asarray(widths, dtype=float)
        raw = pdf_1d_numba(n, float(sigma), float(self.p), widths, int(self.numba_Npts))
        raw = np.maximum(raw, 0.0)
        widths_shifted, pdf_shifted = self._shift_pdf(widths, raw, self.C)
        if self.dC > 0 and len(widths_shifted) > 1:
            pixel_scale = widths_shifted[1] - widths_shifted[0]
            sm_sigma = self.dC / pixel_scale
            if sm_sigma > 0:
                pdf_shifted = gaussian_filter1d(pdf_shifted, sm_sigma, mode='constant', cval=0.0)
        area = np.trapz(pdf_shifted, widths_shifted)
        if area > 0:
            pdf_shifted /= area
        return widths_shifted, pdf_shifted

    # ------------------------
    # 2D PDF plotting (drift time on x-axis)
    # ------------------------
    def plot_pdf_2d(self, n: int, drift_times: Sequence[float], widths: np.ndarray = None,
                    num_points: int = 1000, cmap: str = "viridis", show: bool = True):
        if widths is None:
            lim = 10 * self.M * np.sqrt(max(drift_times))
            widths = np.linspace(0.0, lim, num_points)
        widths = np.asarray(widths, dtype=float)
        pdf2d = np.zeros((len(widths), len(drift_times)))
        for j, dt in enumerate(drift_times):
            _, pdf_slice = self.pdf_1d_numba_wrapper(n, dt, widths=widths)
            pdf2d[:, j] = pdf_slice
        if show:
            plt.figure()
            plt.imshow(pdf2d, aspect="auto", origin="lower",
                       extent=[drift_times[0], drift_times[-1], widths[0], widths[-1]],
                       cmap=cmap)
            plt.colorbar(label="Probability density")
            plt.xlabel("Drift time")
            plt.ylabel("Width")
            plt.title(f"2D Width PDF at N={n}")
        return drift_times, widths, pdf2d

    # ------------------------
    # 1D PDF plotting (numba)
    # ------------------------
    def plot_pdf_1d_numba(self, n: int, drift_time: float, widths: np.ndarray = None, num_points: int = 1000,
                           show: bool = True, ax=None, label: str = None, **kwargs):
        widths_shifted, pdf_shifted = self.pdf_1d_numba_wrapper(n, drift_time, widths=widths, num_points=num_points)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(widths_shifted, pdf_shifted, label=(label if label is not None else f"N={n}, t={drift_time:.2f} (numba)"), **kwargs)
        ax.set_xlabel("Width")
        ax.set_ylabel("PDF")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if show:
            plt.show()
        return widths_shifted, pdf_shifted

    # ------------------------
    # 3D PDF building
    # ------------------------
    def build_pdf_3d(self, n_values: Sequence[int], drift_times: Sequence[float], widths: np.ndarray = None,
                     num_points: int = 1000, show_progress: bool = True) -> None:
        if widths is None:
            max_sigma = self.M * np.sqrt(max(drift_times))
            lim = 10.0 * max_sigma
            widths = np.linspace(0.0, lim, num_points)
        widths = np.asarray(widths, dtype=float)
        pdf3d = np.zeros((len(n_values), len(drift_times), len(widths)))
        iterable = enumerate(n_values) if not show_progress else tqdm(enumerate(n_values), total=len(n_values), desc="Building 3D PDF")
        for i, n in iterable:
            for j, dt in enumerate(drift_times):
                _, pdf_slice = self.pdf_1d_numba_wrapper(n, dt, widths=widths)
                pdf3d[i, j, :] = pdf_slice
        self.pdf_3d = pdf3d
        self._pdf_n_values = np.array(n_values)
        self._pdf_drift_times = np.array(drift_times)
        self._pdf_widths = widths

    # ------------------------
    # Sampling from stored 3D PDF
    # ------------------------
    def sample(self, Ns: np.ndarray, DTs: np.ndarray) -> np.ndarray:
        if not hasattr(self, "pdf_3d"):
            raise RuntimeError("3D PDF not built yet. Call build_pdf_3d first.")
        interpolator = RegularGridInterpolator(
            (self._pdf_n_values, self._pdf_drift_times, self._pdf_widths),
            self.pdf_3d,
            bounds_error=False,
            fill_value=0.0
        )
        Ws = np.zeros(len(Ns))
        for i, (n, dt) in enumerate(zip(Ns, DTs)):
            pdf_slice = interpolator(np.array([[n, dt, w] for w in self._pdf_widths]))
            pdf_slice = np.maximum(pdf_slice, 0.0)
            if np.sum(pdf_slice) == 0:
                pdf_slice = np.ones_like(pdf_slice) / len(pdf_slice)
            else:
                pdf_slice /= np.sum(pdf_slice)
            cdf = np.cumsum(pdf_slice)
            r = np.random.rand()
            Ws[i] = np.interp(r, cdf, self._pdf_widths)
        return Ws
    
    @staticmethod
    def compute_percentile_bounds(pdf_3d: np.ndarray, widths: np.ndarray, lower: float = 0.5, upper: float = 99.5) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute lower and upper percentile bounds along the width axis for each (N, drift_time) in a 3D PDF.

        Args:
            pdf_3d: 3D PDF of shape (num_N, num_drift_times, num_widths)
            widths: 1D array of width values corresponding to the last axis of pdf_3d
            lower: lower percentile (e.g., 1.0 for 1%)
            upper: upper percentile (e.g., 99.0 for 99%)

        Returns:
            lower_bounds: 2D array of shape (num_N, num_drift_times)
            upper_bounds: 2D array of shape (num_N, num_drift_times)
        """
        N_len, dt_len, _ = pdf_3d.shape
        lower_bounds = np.zeros((N_len, dt_len))
        upper_bounds = np.zeros((N_len, dt_len))

        for i in range(N_len):
            for j in range(dt_len):
                cdf = np.cumsum(pdf_3d[i, j, :])
                if cdf[-1] == 0:
                    # fallback if PDF is zero everywhere
                    lower_bounds[i, j] = widths[0]
                    upper_bounds[i, j] = widths[-1]
                    continue
                cdf /= cdf[-1]  # normalize to 1
                lower_bounds[i, j] = np.interp(lower / 100.0, cdf, widths)
                upper_bounds[i, j] = np.interp(upper / 100.0, cdf, widths)

        return lower_bounds, upper_bounds