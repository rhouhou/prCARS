"""
prcars.methods.kk
------------------------
Kramers-Kronig phase retrieval for CARS spectra.

Theory
------
The measured CARS intensity is

    I(ω) = |χ_NR + χ_R(ω)|²

where χ_NR is the real non-resonant background and χ_R is the complex
resonant susceptibility.  After dividing by I_NR = |χ_NR|² the amplitude

    A(ω) = sqrt(I / I_NR)

satisfies the KK dispersion relation: its phase φ(ω) is the Hilbert
transform of its log-amplitude:

    φ(ω) = H[ln A](ω)

and then

    Im[χ_R](ω) ∝ A(ω) · sin(φ(ω))    ← the Raman-equivalent spectrum
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy import signal, interpolate
from scipy.optimize import minimize


class KramersKronig:
    """
    Kramers-Kronig spectral retrieval.

    Parameters
    ----------
    fft_method : bool
        Use the FFT-based Hilbert transform (fast, default).
        Set to ``False`` for the direct principal-value integral (slow,
        but exact and useful for validation on small spectra).
    zero_pad_factor : int
        Zero-padding multiplier when using the FFT method.
        Larger values reduce circular-convolution artefacts. Default 4.
    phase_offset : float
        Manual constant phase added after retrieval (radians). Default 0.
    """

    def __init__(
        self,
        *,
        fft_method: bool = True,
        zero_pad_factor: int = 4,
        phase_offset: float = 0.0,
    ):
        self.fft_method = fft_method
        self.zero_pad_factor = zero_pad_factor
        self.phase_offset = phase_offset

    # ── public interface ──────────────────────────────────────────────────────
    def retrieve(
        self,
        wavenumbers: np.ndarray,
        intensity: np.ndarray,
        *,
        nr_background: np.ndarray | None = None,
    ) -> dict:
        """
        Run KK retrieval and return a raw result dictionary.

        Called internally by :class:`~prcars.pipeline.Pipeline`.
        You can also call it directly.

        Parameters
        ----------
        wavenumbers : 1-D array
            Raman shift axis (cm⁻¹); need not be uniform.
        intensity : 1-D array
            Background-corrected CARS intensity.
        nr_background : 1-D array, optional
            Non-resonant background.  When supplied the ratio I/I_NR is
            formed before the phase retrieval, which suppresses the NR
            spectral envelope.

        Returns
        -------
        dict
            Keys: ``wavenumbers``, ``amplitude``, ``phase``,
            ``im_chi3``, ``re_chi3``.
        """
        wn, I = self._to_uniform(wavenumbers, intensity)
        I = np.maximum(I, 0.0)

        if nr_background is not None:
            _, bg_u = self._to_uniform(wavenumbers, nr_background)
            ratio = I / np.maximum(bg_u, 1e-30)
        else:
            ratio = I

        # Clip wildly small values that cause log blow-up
        ratio_safe = np.maximum(ratio, 1e-6 * np.maximum(ratio.max(), 1e-30))
        amplitude = np.sqrt(ratio_safe)

        # Mean-subtract log amplitude (standard KK-CARS pre-conditioning).
        # The constant offset is absorbed into the phase offset correction later.
        log_amp = np.log(amplitude)
        log_amp -= log_amp.mean()

        if self.fft_method:
            phase = self._phase_fft(log_amp)
        else:
            phase = self._phase_direct(wn, log_amp)

        phase += self.phase_offset

        return {
            "wavenumbers": wn,
            "amplitude":   amplitude,
            "phase":       phase,
            "im_chi3":     amplitude * np.sin(phase),
            "re_chi3":     amplitude * np.cos(phase) - 1.0,
        }

    # ── internals ─────────────────────────────────────────────────────────────
    @staticmethod
    def _to_uniform(wn: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        diffs = np.diff(wn)
        if np.std(diffs) / (np.abs(np.mean(diffs)) + 1e-30) < 1e-3:
            return wn, y
        warnings.warn(
            "KK: wavenumber axis is not uniform – interpolating to uniform grid.",
            stacklevel=3,
        )
        wn_u = np.linspace(wn[0], wn[-1], len(wn))
        y_u = interpolate.interp1d(
            wn, y, bounds_error=False, fill_value="extrapolate"
        )(wn_u)
        return wn_u, y_u

    def _phase_fft(self, log_amplitude: np.ndarray) -> np.ndarray:
        """FFT-based Hilbert transform (O(N log N))."""
        n = len(log_amplitude)
        n_pad = n * self.zero_pad_factor
        padded = np.zeros(n_pad)
        padded[:n] = log_amplitude
        analytic = signal.hilbert(padded)
        return -np.imag(analytic)[:n]

    @staticmethod
    def _phase_direct(wn: np.ndarray, log_amplitude: np.ndarray) -> np.ndarray:
        """Direct principal-value KK integral (O(N²), for validation)."""
        n   = len(wn)
        dw  = wn[1] - wn[0]
        phi = np.zeros(n)
        for i, w0 in enumerate(wn):
            integrand = np.where(
                np.abs(wn - w0) > 0.5 * dw,
                wn * log_amplitude / (wn ** 2 - w0 ** 2 + 1e-30),
                0.0,
            )
            phi[i] = (2.0 / np.pi) * np.trapezoid(integrand, wn)
        return phi

    def __repr__(self) -> str:
        m = "fft" if self.fft_method else "direct"
        return (
            f"KramersKronig(method={m!r}, "
            f"zero_pad_factor={self.zero_pad_factor})"
        )
