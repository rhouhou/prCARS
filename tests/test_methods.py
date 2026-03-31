"""
tests/test_methods.py
---------------------
Extended tests for individual retrieval methods.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import prcars as ca
from prcars.utils import synthetic_cars, spectral_pearson, spectral_mse


# ── KK edge cases ─────────────────────────────────────────────────────────────

class TestKKEdgeCases:
    def test_direct_method(self):
        """Direct KK on a small array should not diverge."""
        wn = np.linspace(2800, 3000, 64)
        cars, _, _ = synthetic_cars(wavenumbers=wn, seed=1)
        kk = ca.KramersKronig(fft_method=False)
        result = kk.retrieve(wn, cars)
        assert np.all(np.isfinite(result["im_chi3"]))

    def test_nonuniform_axis_warns(self):
        """Non-uniform wavenumber axis should trigger a warning."""
        wn = np.concatenate([np.linspace(2700, 2900, 100),
                             np.linspace(2905, 3100, 100)])
        _, cars, _ = synthetic_cars(wavenumbers=np.linspace(2700, 3100, 200))
        kk = ca.KramersKronig()
        with pytest.warns(UserWarning, match="uniform"):
            kk.retrieve(wn, cars)

    def test_zero_pad_factor(self):
        wn, cars, _ = synthetic_cars()
        kk4 = ca.KramersKronig(zero_pad_factor=4)
        kk8 = ca.KramersKronig(zero_pad_factor=8)
        r4 = kk4.retrieve(wn, cars)
        r8 = kk8.retrieve(wn, cars)
        # Both should be finite; higher padding ≠ worse
        assert np.all(np.isfinite(r4["phase"]))
        assert np.all(np.isfinite(r8["phase"]))

    def test_manual_phase_offset(self):
        wn, cars, _ = synthetic_cars()
        kk = ca.KramersKronig(phase_offset=np.pi / 4)
        result = kk.retrieve(wn, cars)
        kk0 = ca.KramersKronig(phase_offset=0.0)
        result0 = kk0.retrieve(wn, cars)
        # Phases should differ by π/4 on average
        diff = np.mean(result["phase"] - result0["phase"])
        assert abs(diff - np.pi / 4) < 0.05

    def test_with_nr_background(self):
        wn, cars, _ = synthetic_cars()
        nr_bg = np.ones_like(cars) * cars.mean() * 0.3
        kk = ca.KramersKronig()
        result = kk.retrieve(wn, cars, nr_background=nr_bg)
        assert np.all(np.isfinite(result["im_chi3"]))


# ── MEM edge cases ────────────────────────────────────────────────────────────

class TestMEMEdgeCases:
    def test_auto_order(self):
        wn, cars, _ = synthetic_cars()
        mem = ca.MaximumEntropy(order=None)   # should auto-select
        result = mem.retrieve(wn, cars)
        assert np.all(np.isfinite(result["im_chi3"]))

    def test_mem_phase_method(self):
        wn, cars, _ = synthetic_cars()
        mem = ca.MaximumEntropy(phase_method="mem_phase", order=64)
        result = mem.retrieve(wn, cars)
        assert result["im_chi3"].shape == wn.shape

    def test_order_clamped_to_n_minus_1(self):
        """Order larger than spectrum length should be silently clamped."""
        wn = np.linspace(2700, 3100, 50)
        _, cars, _ = synthetic_cars(wavenumbers=wn)
        mem = ca.MaximumEntropy(order=9999)
        result = mem.retrieve(wn, cars)
        assert np.all(np.isfinite(result["im_chi3"]))

    def test_correlation_with_truth(self):
        wn, cars, im_true = synthetic_cars(seed=7)
        result = ca.retrieve(wn, cars, method="mem",
                             background="none", auto_phase=True,
                             silent_region=(2700, 2730),
                             retriever_kw={"order": 128})
        r = spectral_pearson(result.im_chi3, im_true)
        assert r > 0.5, f"MEM Pearson correlation too low: {r:.3f}"


# ── NN stubs (no weights needed) ─────────────────────────────────────────────

class TestNNRetriever:
    def test_no_weights_runs(self):
        """Random-weight network should still produce finite output."""
        wn, cars, _ = synthetic_cars()
        nn = ca.NeuralNetRetriever(
            model_name=None,
            architecture="cnn",
            input_len=512,
        )
        result = nn.retrieve(wn, cars)
        assert result["im_chi3"].shape == wn.shape
        assert np.all(np.isfinite(result["im_chi3"]))

    def test_unet_architecture(self):
        wn, cars, _ = synthetic_cars()
        nn = ca.NeuralNetRetriever(
            model_name=None,
            architecture="unet",
            input_len=512,
        )
        result = nn.retrieve(wn, cars)
        assert result["im_chi3"].shape == wn.shape

    def test_pipeline_nn(self):
        wn, cars, _ = synthetic_cars()
        p = ca.Pipeline(
            method="nn",
            background="als",
            retriever_kw={"model_name": None, "architecture": "cnn",
                          "input_len": 512},
        )
        result = p.run(wn, cars)
        assert isinstance(result, ca.CARSResult)

    def test_missing_model_warns(self):
        """Nonexistent bundled model name should warn, not raise."""
        with pytest.warns(UserWarning):
            nn = ca.NeuralNetRetriever(model_name="cars_unet_v1",
                                       architecture="cnn", input_len=512)

    def test_save_load(self, tmp_path):
        wn, cars, _ = synthetic_cars()
        nn = ca.NeuralNetRetriever(model_name=None, architecture="cnn",
                                   input_len=512)
        path = tmp_path / "test_model.pt"
        nn.save(str(path))
        assert path.exists()
        # Reload and run
        nn2 = ca.NeuralNetRetriever(model_name=None, architecture="cnn",
                                    model_path=path, input_len=512)
        result = nn2.retrieve(wn, cars)
        assert np.all(np.isfinite(result["im_chi3"]))


# ── auto_phase_correction ─────────────────────────────────────────────────────

class TestAutoPhase:
    def test_small_silent_region_warns(self):
        wn = np.linspace(2700, 3100, 512)
        im = np.random.randn(512)
        re = np.random.randn(512)
        with pytest.warns(UserWarning, match="fewer than 3"):
            ca.auto_phase_correction(wn, im, re, silent_region=(2700, 2700.5))

    def test_returns_arrays_same_shape(self):
        wn = np.linspace(2700, 3100, 512)
        im = np.random.randn(512)
        re = np.random.randn(512)
        im_c, re_c, phi = ca.auto_phase_correction(wn, im, re,
                                                    silent_region=(2700, 2730))
        assert im_c.shape == im.shape
        assert re_c.shape == re.shape
        assert isinstance(phi, float)

    def test_phase_reduces_silent_signal(self):
        """After correction Im[χ³] should be smaller in the silent region."""
        wn, cars, _ = synthetic_cars()
        result = ca.retrieve(wn, cars, method="kk",
                             background="als", auto_phase=False)
        mask = (wn >= 2700) & (wn <= 2730)
        energy_before = np.sum(result.im_chi3[mask] ** 2)

        result2 = ca.retrieve(wn, cars, method="kk",
                              background="als", auto_phase=True,
                              silent_region=(2700, 2730))
        energy_after = np.sum(result2.im_chi3[mask] ** 2)
        assert energy_after <= energy_before * 1.1  # at most marginal increase


# ── utils ─────────────────────────────────────────────────────────────────────

class TestUtils:
    def test_synthetic_cars_shape(self):
        wn, cars, im = synthetic_cars()
        assert wn.shape == cars.shape == im.shape

    def test_synthetic_cars_positive(self):
        _, cars, _ = synthetic_cars()
        assert np.all(cars >= 0)

    def test_spectral_pearson_identical(self):
        x = np.random.randn(200)
        assert abs(spectral_pearson(x, x) - 1.0) < 1e-10

    def test_spectral_pearson_anticorrelated(self):
        x = np.random.randn(200)
        assert spectral_pearson(x, -x) < -0.99

    def test_spectral_mse_identical(self):
        x = np.random.randn(200)
        assert spectral_mse(x, x) < 1e-10

    def test_benchmark_runs(self):
        wn, cars, im_true = synthetic_cars(seed=99)
        scores = ca.utils.benchmark(wn, cars, im_true, methods=["kk"])
        assert "kk" in scores
        assert "mse" in scores["kk"]
        assert "pearson" in scores["kk"]

    def test_list_models(self):
        from prcars.methods.nn import list_models
        models = list_models()
        assert isinstance(models, list)
        assert "cars_unet_v1" in models
