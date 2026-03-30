"""
tests/conftest.py
-----------------
Shared pytest fixtures for prcars test suite.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def wavenumbers():
    return np.linspace(2700, 3100, 512)


@pytest.fixture(scope="session")
def synthetic_data(wavenumbers):
    from prcars.utils import synthetic_cars
    wn, cars, im_true = synthetic_cars(wavenumbers=wavenumbers, seed=42)
    return wn, cars, im_true


@pytest.fixture(scope="session")
def simple_pipeline():
    import prcars as ca
    return ca.Pipeline(
        method="kk",
        background="als",
        correction="divide",
        denoise="savgol",
        auto_phase=True,
        silent_region=(2700, 2730),
    )
