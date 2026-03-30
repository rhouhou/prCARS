"""
prcars.result
--------------------
Unified result container returned by every retrieval method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CARSResult:
    """
    Container for CARS retrieval output.

    Attributes
    ----------
    wavenumbers : ndarray
        Raman shift axis (cm⁻¹).
    im_chi3 : ndarray
        Imaginary part of χ⁽³⁾ – the Raman-like spectrum.
    re_chi3 : ndarray
        Real part of χ⁽³⁾.
    chi3_complex : ndarray
        Complex χ⁽³⁾ = re_chi3 + 1j * im_chi3.
    amplitude : ndarray
        Spectral amplitude |A(ω)| before phase retrieval.
    phase : ndarray
        Retrieved phase φ(ω) in radians.
    background : ndarray or None
        Estimated non-resonant background.
    phase_offset : float
        Constant phase correction applied (radians).
    method : str
        Retrieval method used ('kk', 'mem', 'nn').
    intermediate : dict
        Intermediate arrays from each pipeline step.
    metadata : dict
        Free-form metadata (model path, hyperparameters, …).
    """

    wavenumbers:  np.ndarray
    im_chi3:      np.ndarray
    re_chi3:      np.ndarray
    chi3_complex: np.ndarray
    amplitude:    np.ndarray
    phase:        np.ndarray
    background:   Optional[np.ndarray] = None
    phase_offset: float                = 0.0
    method:       str                  = "unknown"
    intermediate: dict                 = field(default_factory=dict)
    metadata:     dict                 = field(default_factory=dict)

    # ── convenience ───────────────────────────────────────────────────────────
    def peak_positions(self, prominence: float = 0.1) -> np.ndarray:
        """Return wavenumbers of peaks in Im[χ³] above *prominence*."""
        from scipy.signal import find_peaks
        sig = self.im_chi3
        norm = (sig - sig.min()) / (sig.max() - sig.min() + 1e-30)
        idx, _ = find_peaks(norm, prominence=prominence)
        return self.wavenumbers[idx]

    def normalise(self, method: str = "max") -> "CARSResult":
        """Return a copy with Im[χ³] normalised to [0, 1]."""
        import copy
        out = copy.copy(self)
        if method == "max":
            scale = out.im_chi3.max()
        elif method == "area":
            scale = np.trapezoid(np.abs(out.im_chi3), out.wavenumbers)
        else:
            raise ValueError(f"Unknown normalisation '{method}'.")
        scale = max(scale, 1e-30)
        out.im_chi3 = out.im_chi3 / scale
        return out

    def save(self, path: str) -> None:
        """Save result arrays to a compressed .npz file."""
        np.savez_compressed(
            path,
            wavenumbers  = self.wavenumbers,
            im_chi3      = self.im_chi3,
            re_chi3      = self.re_chi3,
            amplitude    = self.amplitude,
            phase        = self.phase,
            background   = self.background if self.background is not None
                           else np.array([]),
            phase_offset = np.array([self.phase_offset]),
        )

    @classmethod
    def load(cls, path: str) -> "CARSResult":
        """Load a previously saved .npz result."""
        d = np.load(path)
        bg = d["background"] if d["background"].size else None
        return cls(
            wavenumbers  = d["wavenumbers"],
            im_chi3      = d["im_chi3"],
            re_chi3      = d["re_chi3"],
            chi3_complex = d["re_chi3"] + 1j * d["im_chi3"],
            amplitude    = d["amplitude"],
            phase        = d["phase"],
            background   = bg,
            phase_offset = float(d["phase_offset"][0]),
        )

    def plot(self, show: bool = True) -> None:
        """Quick diagnostic plot (requires matplotlib)."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        wn = self.wavenumbers

        # panel 1 – raw background
        if self.background is not None:
            axes[0].plot(wn, self.background, color="tomato", lw=1.5,
                         label="Estimated BG")
        axes[0].set_title("Non-resonant background")
        axes[0].set_ylabel("Intensity (a.u.)")
        axes[0].legend()

        # panel 2 – amplitude
        axes[1].plot(wn, self.amplitude, color="steelblue", lw=1.2,
                     label="Amplitude |A(ω)|")
        axes[1].set_title("Spectral amplitude after BG correction")
        axes[1].set_ylabel("Amplitude (a.u.)")
        axes[1].legend()

        # panel 3 – χ³
        axes[2].plot(wn, self.im_chi3, color="seagreen", lw=1.6,
                     label=r"Im[$\chi^{(3)}$]  ← Raman-like")
        axes[2].plot(wn, self.re_chi3, color="mediumpurple", lw=1.1,
                     ls="--", label=r"Re[$\chi^{(3)}$]")
        axes[2].set_title(f"KK retrieval result  (method={self.method})")
        axes[2].set_ylabel(r"$\chi^{(3)}$ (a.u.)")
        axes[2].set_xlabel(r"Raman shift (cm$^{-1}$)")
        axes[2].legend()

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes
