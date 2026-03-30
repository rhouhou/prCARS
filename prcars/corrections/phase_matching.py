"""prcars.corrections.phase_matching"""
from __future__ import annotations
import numpy as np


def phase_matching_correction(
    wavenumbers: np.ndarray,
    intensity: np.ndarray,
    *,
    delta_k: float = 0.0,
    interaction_length: float = 1.0,
    center_wavenumber: float | None = None,
) -> np.ndarray:
    """
    Divide the CARS spectrum by the sinc² phase-matching envelope.

    For collinear, tightly-focused geometries the envelope is nearly flat
    (``delta_k=0``).  Use non-zero ``delta_k`` for loosely-focused or
    non-collinear setups.

    Parameters
    ----------
    delta_k : float
        Wave-vector mismatch per wavenumber unit (cm).
    interaction_length : float
        Interaction length L (same units as 1/delta_k).
    center_wavenumber : float, optional
        Reference point where Δk = 0; defaults to axis midpoint.
    """
    if center_wavenumber is None:
        center_wavenumber = 0.5 * (wavenumbers[0] + wavenumbers[-1])

    arg = 0.5 * delta_k * (wavenumbers - center_wavenumber) * interaction_length
    envelope = np.sinc(arg / np.pi) ** 2        # np.sinc uses normalised sinc
    envelope = np.where(envelope < 1e-10, 1e-10, envelope)
    return intensity / envelope
