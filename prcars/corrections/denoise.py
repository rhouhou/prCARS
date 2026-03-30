"""prcars.corrections.denoise"""
from __future__ import annotations
import warnings
import numpy as np
from scipy import signal


def denoise_savgol(
    intensity: np.ndarray,
    *,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Savitzky-Golay smoothing (preserves peak positions and widths)."""
    wl = window_length | 1           # ensure odd
    wl = max(wl, polyorder + 2)
    return signal.savgol_filter(intensity, wl, polyorder)


def denoise_wiener(
    intensity: np.ndarray,
    *,
    window: int = 11,
) -> np.ndarray:
    """Wiener adaptive filter (local noise-variance estimation)."""
    return signal.wiener(intensity, mysize=window)


def denoise_wavelet(
    intensity: np.ndarray,
    *,
    wavelet: str = "db4",
    level: int | None = None,
    threshold_mode: str = "soft",
    sigma: float | None = None,
) -> np.ndarray:
    """
    Wavelet shrinkage denoising.

    Requires PyWavelets (``pip install PyWavelets``).

    Parameters
    ----------
    wavelet : str
        Mother wavelet (``'db4'``, ``'sym5'``, ``'coif3'``, …).
    level : int, optional
        Decomposition depth; defaults to maximum.
    threshold_mode : {'soft', 'hard'}
    sigma : float, optional
        Noise standard deviation.  If ``None``, estimated via the MAD of the
        finest-scale detail coefficients.
    """
    try:
        import pywt
    except ImportError as e:
        raise ImportError(
            "Wavelet denoising requires PyWavelets: pip install PyWavelets"
        ) from e

    coeffs = pywt.wavedec(intensity, wavelet, level=level)
    if sigma is None:
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(intensity)))
    new_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode=threshold_mode) for c in coeffs[1:]
    ]
    return pywt.waverec(new_coeffs, wavelet)[: len(intensity)]
