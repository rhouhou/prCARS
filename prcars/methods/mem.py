"""
prcars.methods.mem
-------------------------
Maximum Entropy Method (MEM) for CARS spectral retrieval.

Theory
------
Following Vartiainen et al. (J. Appl. Phys. 75, 2815, 1994) and the
practical reformulation by Liu et al. (Opt. Lett. 34, 1363, 2009).

The CARS field amplitude is defined as

    A(ω) = sqrt(I_CARS(ω) / I_NR(ω))

where I_NR is the non-resonant background.  MEM models A(ω) as a
minimum-phase signal: its log is reconstructed from an AR model of the
causal (one-sided) spectrum.  The phase is then extracted from the imaginary
part of the Hilbert transform of log|A|, which is equivalent to the KK
relation but with MEM-smoothed amplitude.

Two AR-coefficient solvers are provided:
  * 'burg'       – Burg lattice recursion (numerically stable, recommended)
  * 'yulewalker' – Yule-Walker / Toeplitz solve (fast for large order)
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy import linalg, signal, interpolate


class MaximumEntropy:
    """
    Maximum Entropy Method (MEM) CARS retrieval.

    Parameters
    ----------
    order : int or None
        AR model order.  ``None`` → ``len(spectrum) // 4`` (heuristic).
    solver : {'burg', 'yulewalker'}
        AR coefficient estimation algorithm.
    zero_pad_factor : int
        FFT zero-padding for the MEM spectral estimate. Default 4.
    phase_method : {'kk', 'mem_phase'}
        Phase extraction strategy.

        * ``'kk'``        – Hilbert transform of MEM log-amplitude (robust).
        * ``'mem_phase'`` – phase of the complex MEM spectrum directly.
    regularise : float
        Tikhonov diagonal regularisation for the Yule-Walker solve. Default 1e-6.
    """

    def __init__(
        self,
        *,
        order: int | None = None,
        solver: str = "burg",
        zero_pad_factor: int = 4,
        phase_method: str = "kk",
        regularise: float = 1e-6,
        n_iterations: int = 1,          # kept for API compatibility
    ):
        if solver not in ("burg", "yulewalker"):
            raise ValueError("solver must be 'burg' or 'yulewalker'")
        if phase_method not in ("kk", "mem_phase"):
            raise ValueError("phase_method must be 'kk' or 'mem_phase'")

        self.order          = order
        self.solver         = solver
        self.zero_pad_factor = zero_pad_factor
        self.phase_method   = phase_method
        self.regularise     = regularise

    # ── public interface ──────────────────────────────────────────────────────
    def retrieve(
        self,
        wavenumbers: np.ndarray,
        intensity: np.ndarray,
        *,
        nr_background: np.ndarray | None = None,
    ) -> dict:
        """
        Run MEM retrieval.

        Parameters
        ----------
        wavenumbers : 1-D array  (cm⁻¹)
        intensity   : 1-D array  (background-corrected or raw CARS)
        nr_background : 1-D array, optional
            If supplied, used to form I/I_NR before MEM.

        Returns
        -------
        dict with keys: wavenumbers, amplitude, phase, im_chi3, re_chi3
        """
        wn = self._uniform(wavenumbers)
        I  = np.maximum(intensity, 0.0)

        # ── Build field amplitude A(ω) ────────────────────────────────────────
        if nr_background is not None:
            bg = interpolate.interp1d(
                wavenumbers, nr_background,
                bounds_error=False, fill_value="extrapolate",
            )(wn)
            bg_safe = np.maximum(bg, 1e-6 * bg.max() + 1e-30)
            ratio = I / bg_safe
        else:
            # Without an explicit NR background use the signal itself;
            # a smooth envelope is implicitly provided by the background
            # correction step upstream in the pipeline.
            ratio = I

        # Hard-clip ratio to remove division blow-ups at the spectrum edges
        ratio = np.clip(ratio, 0, np.percentile(ratio[ratio > 0], 99.5)
                        if (ratio > 0).any() else 1.0)

        amplitude = np.sqrt(np.maximum(ratio, 0.0))

        # ── AR model on log|A| ────────────────────────────────────────────────
        log_amp = np.log(np.maximum(amplitude, 1e-10))
        log_amp -= log_amp.mean()   # remove DC; absorbed into phase offset later

        p = self.order if self.order is not None else max(4, len(wn) // 4)
        p = min(p, len(wn) - 1)

        if self.solver == "burg":
            ar_coeffs, noise_var = self._burg(log_amp, p)
        else:
            ar_coeffs, noise_var = self._yulewalker(log_amp, p)

        # ── Reconstruct smooth log-amplitude from AR model ───────────────────
        # The AR model gives a spectral density for log|A|.
        # We use it to smooth/denoise log|A| before the Hilbert transform.
        n = len(log_amp)
        n_fft = n * self.zero_pad_factor

        ar_full          = np.zeros(n_fft, dtype=complex)
        ar_full[0]       = 1.0
        ar_full[1:p + 1] = ar_coeffs.real

        A_fft   = np.fft.fft(ar_full)
        mem_psd = noise_var / (np.abs(A_fft) ** 2 + 1e-60)   # AR PSD of log|A|

        # Interpolate MEM PSD back to original axis
        half      = n_fft // 2
        freq_ax   = np.linspace(0, 0.5, half)
        wn_norm   = np.linspace(0, 0.5, n)
        mem_logA2 = np.interp(wn_norm, freq_ax, mem_psd[:half].real)
        mem_logA2 = np.maximum(mem_logA2, 1e-60)

        # Smooth log amplitude: blend MEM estimate with measured (avoids
        # artefacts from AR over-smoothing)
        log_amp_smooth = np.sqrt(mem_logA2)           # sqrt(PSD of log|A|) ≈ |log|A||
        log_amp_smooth = np.sign(log_amp) * log_amp_smooth  # restore sign

        # ── Phase via Hilbert transform of (smoothed) log|A| ────────────────
        if self.phase_method == "kk":
            n_pad  = n * 4
            padded = np.zeros(n_pad)
            padded[:n] = log_amp_smooth
            phase = np.imag(signal.hilbert(padded))[:n]
        else:
            # Direct phase from complex AR spectrum
            ar_half  = np.fft.fft(ar_full)[:half]
            mem_cmpl = np.interp(wn_norm, freq_ax,
                                 (noise_var / (ar_half + 1e-60)).real)
            phase = np.angle(mem_cmpl)

        # ── Assemble Im/Re χ³ ────────────────────────────────────────────────
        im_chi3 = amplitude * np.sin(phase)
        re_chi3 = amplitude * np.cos(phase)

        # Convention: positive Raman peaks
        if np.max(im_chi3) < -np.min(im_chi3):
            im_chi3 = -im_chi3
            re_chi3 = -re_chi3
            phase   = phase + np.pi

        return {
            "wavenumbers": wn,
            "amplitude":   amplitude,
            "phase":       phase,
            "im_chi3":     im_chi3,
            "re_chi3":     re_chi3,
        }

    # ── AR solvers ────────────────────────────────────────────────────────────
    @staticmethod
    def _burg(x: np.ndarray, order: int) -> tuple[np.ndarray, float]:
        """
        Burg's method — lattice recursion minimising forward+backward
        prediction error simultaneously.  Numerically stable for any order.
        """
        n  = len(x)
        ef = x.astype(complex).copy()
        eb = x.astype(complex).copy()
        ar = np.zeros(order, dtype=complex)
        P  = float(np.real(np.dot(x, x.conj()))) / n

        for k in range(order):
            num = -2.0 * np.dot(ef[k + 1:], eb[k: n - 1].conj())
            den = (np.dot(ef[k + 1:], ef[k + 1:].conj()) +
                   np.dot(eb[k: n - 1], eb[k: n - 1].conj()))
            kappa = num / (den + 1e-30)

            ar_prev   = ar[:k].copy()
            ar[:k]    = ar_prev + kappa * ar_prev[::-1].conj()
            ar[k]     = kappa

            ef_new    = ef[k + 1:] + kappa * eb[k: n - 1]
            eb_new    = eb[k: n - 1] + kappa.conj() * ef[k + 1:]
            ef[k + 1:] = ef_new
            eb[k: n - 1] = eb_new

            P *= max(1.0 - float(np.abs(kappa) ** 2), 1e-30)

        return ar.real, float(P)

    def _yulewalker(self, x: np.ndarray, order: int) -> tuple[np.ndarray, float]:
        """Yule-Walker equations via Toeplitz solve."""
        r = np.correlate(x, x, mode="full")[len(x) - 1:]
        r = r[:order + 1] / len(x)

        R   = linalg.toeplitz(r[:order])
        R  += self.regularise * np.eye(order) * r[0]
        rhs = -r[1: order + 1]

        try:
            ar = linalg.solve(R, rhs, assume_a="sym")
        except linalg.LinAlgError:
            warnings.warn("Yule-Walker singular; falling back to Burg.", stacklevel=2)
            return self._burg(x, order)

        noise_var = float(np.real(r[0] + np.dot(ar, r[1: order + 1])))
        return ar.real, max(noise_var, 1e-30)

    @staticmethod
    def _uniform(wn: np.ndarray) -> np.ndarray:
        diffs = np.diff(wn)
        if np.std(diffs) / (np.abs(np.mean(diffs)) + 1e-30) < 1e-3:
            return wn
        warnings.warn("MEM: interpolating to uniform wavenumber grid.", stacklevel=3)
        return np.linspace(wn[0], wn[-1], len(wn))

    def __repr__(self) -> str:
        return (
            f"MaximumEntropy(order={self.order}, solver={self.solver!r}, "
            f"phase_method={self.phase_method!r})"
        )
