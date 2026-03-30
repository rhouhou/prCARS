"""
tests/test_pipeline.py
----------------------
Core tests for the cars-analysis package.
Run with:  pytest -v
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import prcars as ca
from prcars.utils import synthetic_cars, spectral_pearson


# ── shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_data():
    wn, cars, im_true = synthetic_cars(seed=0)
    return wn, cars, im_true


# ── corrections ───────────────────────────────────────────────────────────────

class TestBackgroundEstimators:
    def test_als_shape(self, synthetic_data):
        wn, cars, _ = synthetic_data
        bg = ca.background_als(cars)
        assert bg.shape == cars.shape

    def test_polynomial_shape(self, synthetic_data):
        wn, cars, _ = synthetic_data
        bg = ca.background_polynomial(wn, cars, degree=4)
        assert bg.shape == cars.shape

    def test_snip_shape(self, synthetic_data):
        wn, cars, _ = synthetic_data
        bg = ca.background_snip(cars)
        assert bg.shape == cars.shape
        assert np.all(bg >= 0)

    def test_rolling_ball_shape(self, synthetic_data):
        wn, cars, _ = synthetic_data
        bg = ca.background_rolling_ball(cars, radius=30)
        assert bg.shape == cars.shape

    def test_als_below_signal(self, synthetic_data):
        _, cars, _ = synthetic_data
        bg = ca.background_als(cars, p=0.01)
        # ALS background should be below most of the signal
        assert np.mean(bg <= cars) > 0.6


class TestDenoising:
    def test_savgol(self, synthetic_data):
        _, cars, _ = synthetic_data
        out = ca.denoise_savgol(cars, window_length=11, polyorder=3)
        assert out.shape == cars.shape

    def test_wiener(self, synthetic_data):
        _, cars, _ = synthetic_data
        out = ca.denoise_wiener(cars, window=9)
        assert out.shape == cars.shape

    def test_wavelet_import_error(self, synthetic_data):
        """Wavelet raises ImportError gracefully when pywt is missing."""
        import unittest.mock as mock
        _, cars, _ = synthetic_data
        with mock.patch.dict("sys.modules", {"pywt": None}):
            with pytest.raises(ImportError, match="PyWavelets"):
                ca.denoise_wavelet(cars)


class TestPhaseMatching:
    def test_no_correction(self, synthetic_data):
        wn, cars, _ = synthetic_data
        out = ca.phase_matching_correction(wn, cars, delta_k=0.0)
        np.testing.assert_allclose(out, cars)

    def test_shape_preserved(self, synthetic_data):
        wn, cars, _ = synthetic_data
        out = ca.phase_matching_correction(wn, cars, delta_k=0.01,
                                           interaction_length=0.5)
        assert out.shape == cars.shape


# ── retrieval methods ─────────────────────────────────────────────────────────

class TestKramersKronig:
    def test_output_keys(self, synthetic_data):
        wn, cars, _ = synthetic_data
        kk = ca.KramersKronig()
        result = kk.retrieve(wn, cars)
        for k in ("wavenumbers", "amplitude", "phase", "im_chi3", "re_chi3"):
            assert k in result

    def test_shapes(self, synthetic_data):
        wn, cars, _ = synthetic_data
        kk = ca.KramersKronig()
        result = kk.retrieve(wn, cars)
        assert result["im_chi3"].shape == wn.shape

    def test_correlation_with_truth(self, synthetic_data):
        wn, cars, im_true = synthetic_data
        result = ca.retrieve(wn, cars, method="kk",
                             background="als", auto_phase=True,
                             silent_region=(2700, 2730))
        r = spectral_pearson(result.im_chi3, im_true)
        assert r > 0.7, f"KK Pearson correlation too low: {r:.3f}"


class TestMaximumEntropy:
    def test_output_keys(self, synthetic_data):
        wn, cars, _ = synthetic_data
        mem = ca.MaximumEntropy()
        result = mem.retrieve(wn, cars)
        for k in ("wavenumbers", "amplitude", "phase", "im_chi3", "re_chi3"):
            assert k in result

    def test_burg_solver(self, synthetic_data):
        wn, cars, _ = synthetic_data
        mem = ca.MaximumEntropy(solver="burg", order=64)
        result = mem.retrieve(wn, cars)
        assert not np.any(np.isnan(result["im_chi3"]))

    def test_yulewalker_solver(self, synthetic_data):
        wn, cars, _ = synthetic_data
        mem = ca.MaximumEntropy(solver="yulewalker", order=64)
        result = mem.retrieve(wn, cars)
        assert not np.any(np.isnan(result["im_chi3"]))


# ── pipeline ──────────────────────────────────────────────────────────────────

class TestPipeline:
    @pytest.mark.parametrize("method", ["kk", "mem"])
    def test_run_returns_result(self, synthetic_data, method):
        wn, cars, _ = synthetic_data
        p = ca.Pipeline(method=method, background="als")
        result = p.run(wn, cars)
        assert isinstance(result, ca.CARSResult)
        assert result.im_chi3.shape == wn.shape

    @pytest.mark.parametrize("bg", ["als", "polynomial", "snip", "rolling_ball"])
    def test_background_methods(self, synthetic_data, bg):
        wn, cars, _ = synthetic_data
        result = ca.retrieve(wn, cars, background=bg, auto_phase=False)
        assert not np.any(np.isnan(result.im_chi3))

    @pytest.mark.parametrize("denoise", ["savgol", "wiener", "none"])
    def test_denoise_methods(self, synthetic_data, denoise):
        wn, cars, _ = synthetic_data
        result = ca.retrieve(wn, cars, denoise=denoise, auto_phase=False)
        assert not np.any(np.isnan(result.im_chi3))

    @pytest.mark.parametrize("correction", ["divide", "subtract", "sqrt_divide"])
    def test_correction_modes(self, synthetic_data, correction):
        wn, cars, _ = synthetic_data
        result = ca.retrieve(wn, cars, correction=correction, auto_phase=False)
        assert not np.any(np.isnan(result.im_chi3))

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            ca.Pipeline(method="magic")

    def test_invalid_background_raises(self):
        with pytest.raises(ValueError, match="background"):
            ca.Pipeline(background="dark_matter")

    def test_intermediate_keys(self, synthetic_data):
        wn, cars, _ = synthetic_data
        p = ca.Pipeline(method="kk")
        result = p.run(wn, cars)
        assert "after_correction" in result.intermediate
        assert "background" in result.intermediate

    def test_save_load_roundtrip(self, synthetic_data, tmp_path):
        wn, cars, _ = synthetic_data
        result = ca.retrieve(wn, cars, method="kk", auto_phase=False)
        path = tmp_path / "result.npz"
        result.save(str(path))
        loaded = ca.CARSResult.load(str(path))
        np.testing.assert_allclose(loaded.im_chi3, result.im_chi3, rtol=1e-5)


# ── CARSResult helpers ────────────────────────────────────────────────────────

class TestCARSResult:
    def test_peak_positions(self, synthetic_data):
        wn, cars, _ = synthetic_data
        result = ca.retrieve(wn, cars, method="kk")
        peaks = result.peak_positions(prominence=0.05)
        # expect at least one peak in the CH stretch region
        assert any((2800 <= p <= 3000) for p in peaks), \
            f"No CH-stretch peak found; got {peaks}"

    def test_normalise_max(self, synthetic_data):
        wn, cars, _ = synthetic_data
        result = ca.retrieve(wn, cars, method="kk").normalise("max")
        assert abs(result.im_chi3.max() - 1.0) < 1e-6

    def test_normalise_area(self, synthetic_data):
        wn, cars, _ = synthetic_data
        result = ca.retrieve(wn, cars, method="kk").normalise("area")
        area = np.trapz(np.abs(result.im_chi3), result.wavenumbers)
        assert abs(area - 1.0) < 1e-4
