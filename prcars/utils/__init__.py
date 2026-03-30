"""
prcars.utils
-------------------
Helper utilities: synthetic data generation, benchmarking, and plotting.
"""
from __future__ import annotations
import numpy as np


# ── synthetic CARS generator ──────────────────────────────────────────────────

def synthetic_cars(
    wavenumbers: np.ndarray | None = None,
    *,
    peaks: list[dict] | None = None,
    chi_nr: float = 0.3,
    noise_level: float = 0.01,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic CARS spectrum with known ground-truth Im[χ³].

    Parameters
    ----------
    wavenumbers : 1-D array, optional
        Raman shift axis.  Defaults to 512 points from 2700–3100 cm⁻¹.
    peaks : list of dict, optional
        Each dict can contain ``wn0`` (centre), ``gamma`` (half-width),
        ``amp`` (amplitude).  Defaults to two CH-stretch peaks.
    chi_nr : float
        Non-resonant background amplitude. Default 0.3.
    noise_level : float
        Shot-noise level as a fraction of peak intensity. Default 0.01.
    seed : int or None
        Random seed.

    Returns
    -------
    wavenumbers : ndarray
    cars_intensity : ndarray
    im_chi3_true : ndarray   ← ground-truth for validation
    """
    if wavenumbers is None:
        wavenumbers = np.linspace(2700, 3100, 512)

    if peaks is None:
        peaks = [
            {"wn0": 2850, "gamma": 15, "amp": 1.0},   # CH₂ sym. stretch
            {"wn0": 2920, "gamma": 20, "amp": 0.7},   # CH₃ asym. stretch
            {"wn0": 2960, "gamma": 12, "amp": 0.5},   # CH₃ sym. stretch
        ]

    def lorentzian_complex(wn, wn0, gamma, amp):
        # χ_R = A·γ / (ω - ω₀ + iγ)  →  Im[χ_R] = -A·γ² / ((ω-ω₀)²+γ²)
        return amp * gamma / (wn - wn0 + 1j * gamma)

    chi_r = sum(
        lorentzian_complex(wavenumbers, p["wn0"], p["gamma"], p["amp"])
        for p in peaks
    )

    # Sloped NR background
    slope = 0.05 * np.linspace(0, 1, len(wavenumbers))
    chi_total = (chi_nr + slope) + chi_r

    I_cars = np.abs(chi_total) ** 2
    rng = np.random.default_rng(seed)
    I_cars += rng.normal(0, noise_level * I_cars.max(), I_cars.shape)
    I_cars = np.maximum(I_cars, 0.0)

    # Im[chi_R] from A*gamma/(delta + i*gamma) = -A*gamma^2/((delta)^2+gamma^2)
    # We want positive Raman peaks, so take the negative imaginary part
    im_true = -np.imag(chi_r)

    return wavenumbers, I_cars, im_true


# ── metrics ───────────────────────────────────────────────────────────────────

def spectral_mse(predicted: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error between two spectra (both normalised to [0,1])."""
    p = (predicted - predicted.min()) / ((predicted.max() - predicted.min()) + 1e-30)
    t = (target    - target.min())    / ((target.max() - target.min())    + 1e-30)
    return float(np.mean((p - t) ** 2))


def spectral_pearson(predicted: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation coefficient between two spectra."""
    p = predicted - predicted.mean()
    t = target    - target.mean()
    denom = np.sqrt(np.dot(p, p) * np.dot(t, t)) + 1e-30
    return float(np.dot(p, t) / denom)


def benchmark(
    wavenumbers: np.ndarray,
    cars_intensity: np.ndarray,
    im_chi3_true: np.ndarray,
    methods: list[str] | None = None,
    **pipeline_kw,
) -> dict[str, dict]:
    """
    Run multiple retrieval methods and compare against a ground-truth spectrum.

    Parameters
    ----------
    wavenumbers, cars_intensity, im_chi3_true : arrays
    methods : list of str, optional
        Subset of ``['kk', 'mem', 'nn']``.  Default: ``['kk', 'mem']``.
    **pipeline_kw
        Extra kwargs forwarded to :class:`~prcars.pipeline.Pipeline`.

    Returns
    -------
    dict  method → {'result': CARSResult, 'mse': float, 'pearson': float}
    """
    from prcars.pipeline import Pipeline

    if methods is None:
        methods = ["kk", "mem"]

    results = {}
    for m in methods:
        p   = Pipeline(method=m, **pipeline_kw)
        res = p.run(wavenumbers, cars_intensity)
        results[m] = {
            "result":  res,
            "mse":     spectral_mse(res.im_chi3, im_chi3_true),
            "pearson": spectral_pearson(res.im_chi3, im_chi3_true),
        }
    return results


# ── quick-look plot ───────────────────────────────────────────────────────────

def compare_plot(
    wavenumbers: np.ndarray,
    im_chi3_true: np.ndarray,
    benchmark_results: dict,
    *,
    show: bool = True,
):
    """
    Plot Im[χ³] predictions vs ground truth for all benchmarked methods.

    Parameters
    ----------
    benchmark_results : dict
        Output of :func:`benchmark`.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wavenumbers, im_chi3_true / (im_chi3_true.max() + 1e-30),
            "k--", lw=2, label="Ground truth")

    colors = ["steelblue", "tomato", "seagreen", "darkorange"]
    for (method, data), color in zip(benchmark_results.items(), colors):
        im = data["result"].im_chi3
        im = im / (im.max() + 1e-30)
        label = (f"{method.upper()}  "
                 f"(MSE={data['mse']:.4f}, r={data['pearson']:.3f})")
        ax.plot(wavenumbers, im, color=color, lw=1.5, label=label)

    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    ax.set_ylabel(r"Normalised Im[$\chi^{(3)}$]")
    ax.set_title("CARS retrieval comparison")
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
