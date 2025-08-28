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

@njit(cache=True, parallel=True, fastmath=True)
def pdf_1d_numba(n: int, sigma: float, p: float, widths: np.ndarray, Npts: int = 800) -> np.ndarray:
    out = np.zeros_like(widths)
    for i in prange(widths.shape[0]):
        out[i] = max(0.0, integrand_numba(widths[i], n, sigma, p, 0))  # fallback integrand
    return out

@njit
def sample_from_cdfs(cdfs, widths, r):
    M = cdfs.shape[0]
    Ws = np.empty(M)
    for i in range(M):
        idx = 0
        while idx < cdfs.shape[1] and cdfs[i, idx] < r[i]:
            idx += 1
        if idx >= cdfs.shape[1]:
            idx = cdfs.shape[1]-1
        Ws[i] = widths[idx]
    return Ws

# ------------------------
# WidthModel
# ------------------------

class WidthModel:
    def __init__(self, Dl: float, vd: float, C: float = 0.0, dC: float = 0.0, p: float = 0.5, numba_Npts: int = 800):
        self.Dl = Dl
        self.vd = vd
        self.C = C
        self.dC = dC  # base dC; will scale with sqrt(N) in PDF construction
        self.p = p
        self.M = np.sqrt(2*Dl) / vd
        self.numba_Npts = int(numba_Npts)

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
            sm_sigma = self.dC/(np.sqrt(n)*pixel_scale)  # scale by sqrt(N)
            if sm_sigma > 0:
                pdf_shifted = gaussian_filter1d(pdf_shifted, sm_sigma, mode='constant', cval=0.0)
        area = np.trapz(pdf_shifted, widths_shifted)
        if area > 0:
            pdf_shifted /= area
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
            raise ValueError("Model does not have a precomputed pdf_3d. Run build_pdf_3d first.")
        n_values = self._pdf_n_values
        drift_times = self._pdf_drift_times
        widths = self._pdf_widths
        pdf3d = self.pdf_3d
        n_idx = np.searchsorted(n_values, Ns)
        dt_idx = np.searchsorted(drift_times, DTs)
        n_idx = np.clip(n_idx, 0, len(n_values)-1)
        dt_idx = np.clip(dt_idx, 0, len(drift_times)-1)
        pdf_slices = pdf3d[n_idx, dt_idx, :]
        pdf_slices = np.maximum(pdf_slices, 0.0)
        if pdf_slices.ndim == 1:
            pdf_slices = pdf_slices[None, :]
        pdf_slices /= pdf_slices.sum(axis=1, keepdims=True)
        cdfs = np.cumsum(pdf_slices, axis=1)
        r = np.random.rand(len(Ns))
        Ws = sample_from_cdfs(cdfs, widths, r)
        return Ws

    @staticmethod
    def compute_percentile_bounds(pdf_3d: np.ndarray, widths: np.ndarray, lower: float = 0.5, upper: float = 99.5) -> tuple[np.ndarray, np.ndarray]:
        N_len, dt_len, _ = pdf_3d.shape
        lower_bounds = np.zeros((N_len, dt_len))
        upper_bounds = np.zeros((N_len, dt_len))
        for i in range(N_len):
            for j in range(dt_len):
                cdf = np.cumsum(pdf_3d[i, j, :])
                if cdf[-1] == 0:
                    lower_bounds[i, j] = widths[0]
                    upper_bounds[i, j] = widths[-1]
                    continue
                cdf /= cdf[-1]
                lower_bounds[i, j] = np.interp(lower / 100.0, cdf, widths)
                upper_bounds[i, j] = np.interp(upper / 100.0, cdf, widths)
        return lower_bounds, upper_bounds