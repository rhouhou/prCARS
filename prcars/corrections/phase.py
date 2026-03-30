"""prcars.corrections.phase"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.optimize import minimize


def auto_phase_correction(
    wavenumbers: np.ndarray,
    im_chi3: np.ndarray,
    re_chi3: np.ndarray,
    *,
    silent_region: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Find the constant phase offset that minimises the RMS signal in a
    spectroscopically silent region (where Im[χ_R] ≈ 0 by definition).

    The optimisation minimises

        Σ  [sin(φ_orig(ω) + Δφ)]²    for ω in silent_region

    which is equivalent to rotating the complex χ³ until its imaginary
    projection onto the silent window vanishes.

    Parameters
    ----------
    silent_region : (wn_low, wn_high), optional
        Spectral window known to contain no Raman features.
        Defaults to the first 5 % of the wavenumber axis.

    Returns
    -------
    im_corrected, re_corrected, optimal_phase_offset (radians)
    """
    if silent_region is None:
        span = wavenumbers[-1] - wavenumbers[0]
        silent_region = (wavenumbers[0], wavenumbers[0] + 0.05 * span)

    mask = (wavenumbers >= silent_region[0]) & (wavenumbers <= silent_region[1])
    if mask.sum() < 3:
        warnings.warn(
            "auto_phase_correction: silent region contains fewer than 3 points; "
            "skipping phase correction.", stacklevel=2
        )
        return im_chi3, re_chi3, 0.0

    phase_orig = np.arctan2(im_chi3, re_chi3)
    amplitude  = np.sqrt(im_chi3 ** 2 + re_chi3 ** 2)

    # Normalise amplitude in silent region to unit weight so that noise
    # spikes do not dominate the cost function.
    amp_silent = amplitude[mask]
    amp_w      = amp_silent / (amp_silent.max() + 1e-30)   # weights ∈ [0,1]

    def cost(phi):
        residuals = amp_w * np.sin(phase_orig[mask] + phi[0])
        return float(np.dot(residuals, residuals))

    # Global grid search over [−π, π] to avoid local minima
    grid     = np.linspace(-np.pi, np.pi, 73)
    best_phi = min(grid, key=lambda phi: cost([phi]))

    # Local refinement
    res     = minimize(cost, x0=[best_phi], method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-15, "maxiter": 30_000})
    phi_opt = float(res.x[0])

    im_corr = amplitude * np.sin(phase_orig + phi_opt)
    re_corr = amplitude * np.cos(phase_orig + phi_opt)

    # Convention: Raman-like Im[χ³] has positive peaks.
    # Flip globally if the dominant signal excursion is negative.
    if np.max(im_corr) < -np.min(im_corr):
        phi_opt += np.pi
        im_corr  = -im_corr
        re_corr  = -re_corr

    return im_corr, re_corr, phi_opt
