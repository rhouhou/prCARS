"""
examples/quickstart.py
----------------------
Demonstrates the full cars-analysis pipeline on synthetic data.

Run:
    python examples/quickstart.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import prcars as ca
from prcars.utils import synthetic_cars, benchmark, compare_plot

# ── 1. Generate synthetic CARS data ──────────────────────────────────────────
print("=" * 60)
print("  cars-analysis  –  quickstart example")
print("=" * 60)

wn, cars_raw, im_true = synthetic_cars(
    peaks=[
        {"wn0": 2850, "gamma": 15, "amp": 1.0},
        {"wn0": 2920, "gamma": 20, "amp": 0.7},
        {"wn0": 2960, "gamma": 12, "amp": 0.5},
    ],
    chi_nr=0.3,
    noise_level=0.015,
)
print(f"\nSpectrum: {len(wn)} points, {wn[0]:.0f}–{wn[-1]:.0f} cm⁻¹")

# ── 2. One-liner KK retrieval ─────────────────────────────────────────────────
print("\n[1] Kramers-Kronig retrieval (one-liner) …")
result_kk = ca.retrieve(
    wn, cars_raw,
    method       = "kk",
    background   = "als",
    correction   = "divide",
    denoise      = "savgol",
    auto_phase   = True,
    silent_region= (2700, 2730),
)
print(f"    Phase offset : {result_kk.phase_offset:.4f} rad")
print(f"    Peak positions: {result_kk.peak_positions(prominence=0.1)} cm⁻¹")

# ── 3. MEM retrieval ──────────────────────────────────────────────────────────
print("\n[2] Maximum Entropy Method retrieval …")
result_mem = ca.retrieve(
    wn, cars_raw,
    method       = "mem",
    background   = "snip",
    correction   = "divide",
    denoise      = "savgol",
    auto_phase   = True,
    silent_region= (2700, 2730),
    retriever_kw = {"order": 128, "solver": "burg"},
)
print(f"    Peak positions: {result_mem.peak_positions(prominence=0.1)} cm⁻¹")

# ── 4. Pipeline object ────────────────────────────────────────────────────────
print("\n[3] Using Pipeline object (reusable settings) …")
pipeline = ca.Pipeline(
    method        = "kk",
    background    = "als",
    correction    = "divide",
    denoise       = "wiener",
    auto_phase    = True,
    silent_region = (2700, 2730),
    background_kw = {"lam": 5e4, "p": 0.01},
)
result_pipe = pipeline.run(wn, cars_raw)
print(f"    Pipeline: {pipeline}")

# ── 5. Benchmark ──────────────────────────────────────────────────────────────
print("\n[4] Benchmarking KK vs MEM …")
scores = benchmark(
    wn, cars_raw, im_true,
    methods      = ["kk", "mem"],
    background   = "als",
    auto_phase   = True,
    silent_region= (2700, 2730),
)
for method, data in scores.items():
    print(f"    {method.upper():3s}  MSE={data['mse']:.5f}  "
          f"Pearson r={data['pearson']:.4f}")

# ── 6. Save & reload ──────────────────────────────────────────────────────────
print("\n[5] Save / reload result …")
result_kk.save("/tmp/cars_kk_result.npz")
loaded = ca.CARSResult.load("/tmp/cars_kk_result.npz")
assert np.allclose(loaded.im_chi3, result_kk.im_chi3)
print("    Round-trip save/load: OK")

# ── 7. Plot (if matplotlib available) ────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    print("\n[6] Generating comparison plot …")
    compare_plot(wn, im_true, scores, show=False)
    plt.savefig("/tmp/cars_comparison.png", dpi=150)
    print("    Saved to /tmp/cars_comparison.png")

    result_kk.plot(show=False)
    plt.savefig("/tmp/cars_kk_detail.png", dpi=150)
    print("    Detail plot saved to /tmp/cars_kk_detail.png")
except ImportError:
    print("\n    (matplotlib not installed – skipping plots)")

print("\n✓ Quickstart complete.\n")
