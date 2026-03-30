"""
prcars
=============
A unified toolkit for extracting the imaginary part of χ⁽³⁾ from
Coherent Anti-Stokes Raman Scattering (CARS) spectra.

Three retrieval methods are supported
--------------------------------------
* Kramers-Kronig (KK)          – phase retrieval via Hilbert transform
* Maximum Entropy Method (MEM) – Bayesian spectral deconvolution
* Neural Network (NN)          – pretrained / fine-tuneable deep models

Usage
-----
>>> import prcars as ca

>>> # one-liner with defaults
>>> result = ca.retrieve(wavenumbers, cars_intensity, method="kk")
>>> im_chi3 = result.im_chi3

>>> # full pipeline builder
>>> pipeline = ca.Pipeline(method="mem", background="als", denoise="savgol")
>>> result   = pipeline.run(wavenumbers, cars_intensity)
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("prcars")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

# ── public API ────────────────────────────────────────────────────────────────
from prcars.pipeline import Pipeline, retrieve
from prcars.result import CARSResult

from prcars.methods.kk  import KramersKronig
from prcars.methods.mem import MaximumEntropy
from prcars.methods.nn  import NeuralNetRetriever

from prcars.corrections.background import (
    background_als,
    background_polynomial,
    background_snip,
    background_rolling_ball,
)
from prcars.corrections.phase_matching import phase_matching_correction
from prcars.corrections.denoise import (
    denoise_savgol,
    denoise_wiener,
    denoise_wavelet,
)
from prcars.corrections.phase import auto_phase_correction

__all__ = [
    # pipeline
    "Pipeline",
    "retrieve",
    "CARSResult",
    # retrievers
    "KramersKronig",
    "MaximumEntropy",
    "NeuralNetRetriever",
    # corrections
    "background_als",
    "background_polynomial",
    "background_snip",
    "background_rolling_ball",
    "phase_matching_correction",
    "denoise_savgol",
    "denoise_wiener",
    "denoise_wavelet",
    "auto_phase_correction",
]
