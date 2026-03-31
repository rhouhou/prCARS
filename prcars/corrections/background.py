"""
prcars.corrections.background
-------------------------------------
Background estimation strategies and correction modes.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


# ── estimators ────────────────────────────────────────────────────────────────

def background_polynomial(
    wavenumbers: np.ndarray,
    intensity: np.ndarray,
    *,
    degree: int = 5,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Polynomial fit to background-only regions.

    Parameters
    ----------
    degree : int
        Polynomial degree. Default 5.
    mask : boolean array, optional
        True where the spectrum is background-only.
        If ``None``, an automatic off-peak mask is built by keeping points
        below the 30th-percentile threshold.
    """
    if mask is None:
        threshold = np.percentile(intensity, 30)
        mask = intensity <= threshold
    if mask.sum() < degree + 1:
        warnings.warn("Too few background points; falling back to full-spectrum fit.")
        mask = np.ones(len(intensity), dtype=bool)
    coeffs = np.polyfit(wavenumbers[mask], intensity[mask], degree)
    return np.polyval(coeffs, wavenumbers)


def background_snip(
    intensity: np.ndarray,
    *,
    max_iterations: int = 100,
    decreasing: bool = True,
) -> np.ndarray:
    """
    Statistics-sensitive Non-linear Iterative Peak-clipping (SNIP).

    Works in the square-root-of-square-root domain to handle Poisson noise.

    Reference: C.G. Ryan et al., NIM B 34, 396 (1988).

    Parameters
    ----------
    max_iterations : int
        Maximum window half-width to scan. Increase for broader peaks.
    decreasing : bool
        Use decreasing window sequence (recommended, avoids over-subtraction).
    """
    n = len(intensity)
    v = np.sqrt(np.sqrt(np.maximum(intensity, 0.0)))

    iters = range(max_iterations, 0, -1) if decreasing else range(1, max_iterations + 1)
    for w in iters:
        v_new = v.copy()
        for i in range(w, n - w):
            v_new[i] = min(v[i], 0.5 * (v[i - w] + v[i + w]))
        v = v_new

    return v ** 4


def background_als(
    intensity: np.ndarray,
    *,
    lam: float = 1e5,
    p: float = 0.01,
    n_iter: int = 50,
) -> np.ndarray:
    """
    Asymmetric Least Squares (ALS) baseline.

    Parameters
    ----------
    lam : float
        Smoothness penalty (larger → smoother). Default 1e5.
    p : float
        Asymmetry parameter for peaks above the baseline. Default 0.01.
    n_iter : int
        Number of re-weighting iterations. Default 10.

    Reference: P.H.C. Eilers & H.F.M. Boelens (2005).
    """
    n = len(intensity)
    D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), dtype=float).tocsc()
    H = lam * D.T @ D
    w = np.ones(n)
    z = intensity.copy()
    for _ in range(n_iter):
        W = diags(w, 0, dtype=float).tocsc()
        z = spsolve((W + H).tocsc(), w * intensity)
        w = np.where(intensity > z, p, 1 - p)
    return np.maximum(z, 1e-30)


def background_rolling_ball(
    intensity: np.ndarray,
    *,
    radius: int = 50,
) -> np.ndarray:
    """
    Rolling-ball baseline (1-D).

    A sphere of the given radius is rolled beneath the spectrum; the trace
    of its highest contact point defines the background.

    Parameters
    ----------
    radius : int
        Ball radius in data-point units. Default 50.
    """
    n = len(intensity)
    x = np.arange(-radius, radius + 1, dtype=float)
    ball = radius - np.sqrt(np.maximum(radius ** 2 - x ** 2, 0.0))

    bg = np.empty(n)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        b_slice = ball[lo - i + radius: hi - i + radius]
        candidates = intensity[lo:hi] + b_slice
        bg[i] = np.min(candidates) - ball[radius + (lo - i)]

    return np.maximum(bg, 0.0)


# ── correction ────────────────────────────────────────────────────────────────

def background_correction(
    intensity: np.ndarray,
    background: np.ndarray,
    *,
    mode: str = "divide",
    floor: float = 0.0,
) -> np.ndarray:
    """
    Remove the non-resonant background.

    Parameters
    ----------
    mode : {'subtract', 'divide', 'sqrt_divide'}
        * ``'subtract'``    – I − BG  (additive model)
        * ``'divide'``      – I / BG  (multiplicative; maps NR → 1)
        * ``'sqrt_divide'`` – √I / √BG  (amplitude domain)
    floor : float
        Minimum value clamped after correction. Default 0.

    Returns
    -------
    corrected : ndarray
    """
    bg = np.maximum(background, 1e-30)
    if mode == "subtract":
        out = intensity - background
    elif mode == "divide":
        out = intensity / bg
    elif mode == "sqrt_divide":
        out = np.sqrt(np.maximum(intensity, 0.0)) / np.sqrt(bg)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose subtract/divide/sqrt_divide.")
    return np.maximum(out, floor)
