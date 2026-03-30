"""
prcars.pipeline
----------------------
Orchestrates the full pre-processing → retrieval → post-processing chain.
"""
from __future__ import annotations

import warnings
from typing import Optional, Union
import numpy as np

from prcars.result import CARSResult


# ── registry helpers ──────────────────────────────────────────────────────────

_BG_METHODS = ("als", "polynomial", "snip", "rolling_ball", "none")
_DENOISE_METHODS = ("savgol", "wiener", "wavelet", "none")
_RETRIEVAL_METHODS = ("kk", "mem", "nn")
_CORRECTION_MODES = ("subtract", "divide", "sqrt_divide", "none")


class Pipeline:
    """
    Full CARS processing pipeline.

    Parameters
    ----------
    method : {'kk', 'mem', 'nn'}
        Spectral retrieval algorithm.
    background : str or None
        Background estimation method.  One of:
        'als', 'polynomial', 'snip', 'rolling_ball', 'none'.
    correction : str
        How to remove the estimated background:
        'subtract', 'divide', 'sqrt_divide', 'none'.
    denoise : str or None
        Denoising applied *before* background estimation:
        'savgol', 'wiener', 'wavelet', 'none'.
    phase_matching : dict or None
        If given, passed to :func:`~prcars.corrections.phase_matching
        .phase_matching_correction` as keyword arguments.
        Example: ``{'delta_k': 0.02, 'interaction_length': 0.5}``.
    auto_phase : bool
        Whether to apply automatic constant-phase correction after retrieval.
    silent_region : (float, float) or None
        Wavenumber window used for auto-phase correction.
    background_kw : dict
        Extra keyword arguments forwarded to the background estimator.
    denoise_kw : dict
        Extra keyword arguments forwarded to the denoiser.
    retriever_kw : dict
        Extra keyword arguments forwarded to the retrieval method constructor.

    Examples
    --------
    >>> import prcars as ca
    >>> p = ca.Pipeline(method='kk', background='als', denoise='savgol')
    >>> result = p.run(wavenumbers, intensity)
    >>> result.plot()

    >>> # MEM with custom settings
    >>> p = ca.Pipeline(method='mem', background='snip',
    ...                 retriever_kw={'n_iterations': 200})
    >>> result = p.run(wavenumbers, intensity)

    >>> # Neural-network retrieval (pretrained)
    >>> p = ca.Pipeline(method='nn',
    ...                 retriever_kw={'model_name': 'cars_unet_v1'})
    >>> result = p.run(wavenumbers, intensity)
    """

    def __init__(
        self,
        *,
        method: str = "kk",
        background: str = "rolling_ball",
        correction: str = "divide",
        denoise: str = "savgol",
        phase_matching: Optional[dict] = None,
        auto_phase: bool = True,
        silent_region: Optional[tuple] = None,
        background_kw: Optional[dict] = None,
        denoise_kw: Optional[dict] = None,
        retriever_kw: Optional[dict] = None,
    ):
        self.method = method.lower()
        self.background = (background or "none").lower()
        self.correction = correction.lower()
        self.denoise = (denoise or "none").lower()
        self.phase_matching = phase_matching
        self.auto_phase = auto_phase
        self.silent_region = silent_region
        self.background_kw = background_kw or {}
        self.denoise_kw = denoise_kw or {}
        self.retriever_kw = retriever_kw or {}

        self._validate()
        self._retriever = self._build_retriever()

    # ── validation ────────────────────────────────────────────────────────────
    def _validate(self):
        if self.method not in _RETRIEVAL_METHODS:
            raise ValueError(f"method must be one of {_RETRIEVAL_METHODS}, got '{self.method}'")
        if self.background not in _BG_METHODS:
            raise ValueError(f"background must be one of {_BG_METHODS}, got '{self.background}'")
        if self.correction not in _CORRECTION_MODES:
            raise ValueError(f"correction must be one of {_CORRECTION_MODES}, got '{self.correction}'")
        if self.denoise not in _DENOISE_METHODS:
            raise ValueError(f"denoise must be one of {_DENOISE_METHODS}, got '{self.denoise}'")

    # ── retriever factory ─────────────────────────────────────────────────────
    def _build_retriever(self):
        if self.method == "kk":
            from prcars.methods.kk import KramersKronig
            return KramersKronig(**self.retriever_kw)
        elif self.method == "mem":
            from prcars.methods.mem import MaximumEntropy
            return MaximumEntropy(**self.retriever_kw)
        elif self.method == "nn":
            from prcars.methods.nn import NeuralNetRetriever
            return NeuralNetRetriever(**self.retriever_kw)

    # ── individual steps (public for scripting) ───────────────────────────────
    def apply_phase_matching(self, wn: np.ndarray, I: np.ndarray) -> np.ndarray:
        if self.phase_matching is None:
            return I
        from prcars.corrections.phase_matching import phase_matching_correction
        return phase_matching_correction(wn, I, **self.phase_matching)

    def apply_denoise(self, I: np.ndarray) -> np.ndarray:
        if self.denoise == "none":
            return I.copy()
        from prcars.corrections import denoise as _d
        fn = {
            "savgol": _d.denoise_savgol,
            "wiener": _d.denoise_wiener,
            "wavelet": _d.denoise_wavelet,
        }[self.denoise]
        return fn(I, **self.denoise_kw)

    def estimate_background(self, wn: np.ndarray, I: np.ndarray) -> Optional[np.ndarray]:
        if self.background == "none":
            return None
        from prcars.corrections import background as _bg
        fn = {
            "als":          _bg.background_als,
            "polynomial":   _bg.background_polynomial,
            "snip":         _bg.background_snip,
            "rolling_ball": _bg.background_rolling_ball,
        }[self.background]
        # als / polynomial need the wavenumber axis; snip / rolling_ball don't
        if self.background in ("als", "snip", "rolling_ball"):
            return fn(I, **self.background_kw)
        return fn(wn, I, **self.background_kw)

    def apply_correction(
        self, I: np.ndarray, bg: Optional[np.ndarray]
    ) -> np.ndarray:
        if bg is None or self.correction == "none":
            return I
        from prcars.corrections.background import background_correction
        return background_correction(I, bg, mode=self.correction)

    # ── main run ──────────────────────────────────────────────────────────────
    def run(
        self,
        wavenumbers: np.ndarray,
        cars_intensity: np.ndarray,
    ) -> CARSResult:
        """
        Execute the full pipeline and return a :class:`~prcars.result.CARSResult`.

        Parameters
        ----------
        wavenumbers : 1-D array
            Raman shift axis (cm⁻¹).  Does not need to be uniform; the
            retrieval steps will interpolate if necessary.
        cars_intensity : 1-D array
            Raw CARS intensity (same length as *wavenumbers*).

        Returns
        -------
        CARSResult
        """
        wn = np.asarray(wavenumbers, dtype=float)
        I  = np.maximum(np.asarray(cars_intensity, dtype=float), 0.0)
        inter = {}

        # 1 – phase-matching correction
        I = self.apply_phase_matching(wn, I)
        inter["after_phase_matching"] = I.copy()

        # 2 – denoising
        I = self.apply_denoise(I)
        inter["after_denoise"] = I.copy()

        # 3 – background estimation
        bg = self.estimate_background(wn, I)
        inter["background"] = bg

        # 4 – background correction
        I_corr = self.apply_correction(I, bg)
        inter["after_correction"] = I_corr.copy()

        # 5 – retrieval
        raw = self._retriever.retrieve(wn, I_corr)
        inter.update({f"retriever_{k}": v for k, v in raw.items()
                      if k != "wavenumbers"})

        wn_out   = raw["wavenumbers"]
        im_chi3  = raw["im_chi3"]
        re_chi3  = raw["re_chi3"]
        amplitude= raw["amplitude"]
        phase    = raw["phase"]

        # 6 – auto-phase correction
        phi_offset = 0.0
        if self.auto_phase:
            from prcars.corrections.phase import auto_phase_correction
            im_chi3, re_chi3, phi_offset = auto_phase_correction(
                wn_out, im_chi3, re_chi3, silent_region=self.silent_region
            )

        return CARSResult(
            wavenumbers  = wn_out,
            im_chi3      = im_chi3,
            re_chi3      = re_chi3,
            chi3_complex = re_chi3 + 1j * im_chi3,
            amplitude    = amplitude,
            phase        = phase,
            background   = bg,
            phase_offset = phi_offset,
            method       = self.method,
            intermediate = inter,
            metadata     = {"retriever": repr(self._retriever)},
        )

    def __repr__(self) -> str:
        return (
            f"Pipeline(method={self.method!r}, background={self.background!r}, "
            f"correction={self.correction!r}, denoise={self.denoise!r})"
        )


# ── convenience top-level function ────────────────────────────────────────────

def retrieve(
    wavenumbers: np.ndarray,
    cars_intensity: np.ndarray,
    *,
    method: str = "kk",
    background: str = "rolling_ball",
    correction: str = "divide",
    denoise: str = "savgol",
    phase_matching: Optional[dict] = None,
    auto_phase: bool = True,
    silent_region: Optional[tuple] = None,
    background_kw: Optional[dict] = None,
    denoise_kw: Optional[dict] = None,
    retriever_kw: Optional[dict] = None,
) -> CARSResult:
    """
    One-call interface to the full CARS pipeline.

    Parameters
    ----------
    wavenumbers : 1-D array
    cars_intensity : 1-D array
    method : {'kk', 'mem', 'nn'}
        Retrieval algorithm. Default ``'kk'``.
    background : {'als', 'polynomial', 'snip', 'rolling_ball', 'none'}
        Background estimation strategy. Default ``'als'``.
    correction : {'divide', 'subtract', 'sqrt_divide', 'none'}
        How the background is removed. Default ``'divide'``.
    denoise : {'savgol', 'wiener', 'wavelet', 'none'}
        Denoising applied before retrieval. Default ``'savgol'``.
    phase_matching : dict or None
        Phase-matching correction kwargs (``delta_k``, ``interaction_length``).
    auto_phase : bool
        Apply automatic constant-phase offset correction. Default ``True``.
    silent_region : (float, float) or None
        Spectroscopically silent window for phase correction.
    background_kw, denoise_kw, retriever_kw : dict
        Extra keyword arguments for each step.

    Returns
    -------
    CARSResult

    Examples
    --------
    >>> import prcars as ca
    >>> result = ca.retrieve(wn, cars, method='kk')
    >>> result = ca.retrieve(wn, cars, method='mem', background='snip')
    >>> result = ca.retrieve(wn, cars, method='nn',
    ...                      retriever_kw={'model_name': 'cars_unet_v1'})
    """
    return Pipeline(
        method        = method,
        background    = background,
        correction    = correction,
        denoise       = denoise,
        phase_matching= phase_matching,
        auto_phase    = auto_phase,
        silent_region = silent_region,
        background_kw = background_kw or {},
        denoise_kw    = denoise_kw or {},
        retriever_kw  = retriever_kw or {},
    ).run(wavenumbers, cars_intensity)
