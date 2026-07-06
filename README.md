# prCARS

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/status-alpha-orange)
![Project Type](https://img.shields.io/badge/project-scientific%20Python-purple)
![CI](https://github.com/rhouhou/prCARS/actions/workflows/ci.yml/badge.svg)

**prCARS** is a Python toolkit for phase retrieval, non-resonant-background correction, preprocessing, and Raman-like signal reconstruction from Coherent Anti-Stokes Raman Scattering (CARS/BCARS) spectra.

The package provides a modular pipeline for recovering Raman-like information from CARS spectra using classical retrieval methods such as Kramers-Kronig and Maximum Entropy Method, with optional neural-network retrieval support.

---

## Why this project matters

CARS and BCARS spectra contain chemically meaningful Raman-like information, but the measured signal is affected by the non-resonant background, interference effects, instrument response, noise, and baseline artifacts.

prCARS provides a research-oriented Python workflow for:

* preprocessing CARS/BCARS spectra
* estimating and correcting background contributions
* retrieving Raman-like spectral content
* comparing retrieval methods
* testing phase-retrieval pipelines on synthetic examples

The goal is to make CARS/BCARS spectral retrieval easier to test, compare, and integrate into scientific machine-learning workflows.

---

## What this repository demonstrates

This project demonstrates:

* scientific signal-processing package design
* modular spectroscopy preprocessing
* phase-retrieval workflow construction
* non-resonant-background correction
* synthetic CARS/BCARS test data generation
* benchmark utilities for retrieval evaluation
* foundations for integration with CARSBench and CARSGuard

---

## Example output

![prCARS synthetic retrieval example](docs/assets/example_retrieval.png)

---

## Key features

* Kramers-Kronig phase retrieval
* Maximum Entropy Method retrieval
* Optional neural-network retrieval interface
* Background estimation methods:

  * ALS
  * polynomial fitting
  * SNIP
  * rolling-ball
* Background correction methods:

  * subtract
  * divide
  * square-root divide
* Denoising methods:

  * Savitzky-Golay
  * Wiener
  * wavelet-based denoising
* Optional phase-matching correction
* Automatic phase correction with silent-region optimization
* Synthetic CARS example generation
* Benchmark utilities for comparing retrieval results
* Reusable pipeline object for reproducible workflows

---

## Project status

prCARS is currently an **alpha-stage research and portfolio project**.

| Component                          | Status                  |
| ---------------------------------- | ----------------------- |
| Kramers-Kronig retrieval           | Implemented             |
| MEM retrieval                      | Implemented             |
| Background estimation              | Implemented             |
| Background correction              | Implemented             |
| Denoising utilities                | Implemented             |
| Synthetic CARS utility             | Implemented             |
| Benchmark helper utilities         | Implemented             |
| Neural-network retrieval interface | Experimental / optional |
| GitHub Actions CI | Implemented |
| Real-data validation workflow | Planned |
| Integration with CARSBench | Planned |
| Integration with CARSGuard | Planned |
| Full documentation site | Planned |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/rhouhou/prCARS.git
cd prCARS
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On some macOS/Linux systems, you may need to use `python3` instead of `python`.

Install the package in editable mode:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

For plotting helpers:

```bash
python -m pip install -e ".[plot]"
```

For wavelet denoising support:

```bash
python -m pip install -e ".[wavelet]"
```

For optional PyTorch neural-network support:

```bash
python -m pip install -e ".[torch]"
```

For most local development:

```bash
python -m pip install -e ".[dev,plot,wavelet]"
```

---

## Installation check

After installation, verify that the package can be imported:

```bash
python -c "import prcars; print(prcars.__name__)"
```

Expected output:

```text
prcars
```

Run the test suite:

```bash
python -m pytest
```

---

## Quickstart

```python
import numpy as np
import prcars as ca

# Example input arrays
wavenumbers = np.load("wavenumbers.npy")
cars_raw = np.load("cars_spectrum.npy")

# Default retrieval pipeline
result = ca.retrieve(
    wavenumbers,
    cars_raw,
)

im_chi3 = result.im_chi3
print(im_chi3)
```

The recovered `im_chi3` signal represents a Raman-like target extracted from the CARS/BCARS spectrum.

---

## Synthetic example

prCARS includes utilities for generating synthetic CARS-like test data.

```python
from prcars.utils import synthetic_cars

wavenumbers, cars_raw, im_true = synthetic_cars(seed=0)

result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
)

print(result.im_chi3.shape)
```

Synthetic examples are useful for checking whether the retrieval pipeline works before applying it to experimental data.

---

## Retrieval methods

### Kramers-Kronig retrieval

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
)
```

Advanced example:

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
    background="als",
    correction="divide",
    denoise="savgol",
    auto_phase=True,
    silent_region=(2700, 2730),
    retriever_kw={"zero_pad_factor": 8},
)
```

---

### Maximum Entropy Method retrieval

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="mem",
    background="snip",
    correction="divide",
    retriever_kw={
        "order": 128,
        "solver": "burg",
        "phase_method": "kk",
    },
)
```

---

### Optional neural-network retrieval

Neural-network retrieval is optional and requires an additional backend such as PyTorch or TensorFlow.

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="nn",
    background="als",
    retriever_kw={
        "model_name": "cars_unet_v1",
        "backend": "torch",
    },
)
```

This interface is experimental and intended for future learned retrieval workflows.

---

## Pipeline object

For reproducible workflows, use the `Pipeline` object directly.

```python
import prcars as ca

pipeline = ca.Pipeline(
    method="kk",
    background="als",
    correction="divide",
    denoise="savgol",
    auto_phase=True,
    silent_region=(2700, 2730),
)

result_1 = pipeline.run(wavenumbers, spectrum_1)
result_2 = pipeline.run(wavenumbers, spectrum_2)
```

This is useful when applying the same retrieval configuration to multiple spectra.

---

## Background and preprocessing options

| Step                  | Options                                             |
| --------------------- | --------------------------------------------------- |
| Background estimation | `als`, `polynomial`, `snip`, `rolling_ball`, `none` |
| Background correction | `subtract`, `divide`, `sqrt_divide`, `none`         |
| Denoising             | `savgol`, `wiener`, `wavelet`, `none`               |
| Retrieval             | `kk`, `mem`, `nn`                                   |
| Phase correction      | automatic silent-region optimization                |

---

## Benchmarking

prCARS includes utilities for comparing retrieval methods against a known synthetic target.

```python
from prcars.utils import synthetic_cars, benchmark

wavenumbers, cars_raw, im_true = synthetic_cars(seed=0)

scores = benchmark(
    wavenumbers,
    cars_raw,
    im_true,
    methods=["kk", "mem"],
)

for method, values in scores.items():
    print(method, values)
```

These benchmark utilities are intended for quick sanity checks and method comparisons.

---

## Example workflow

A typical retrieval workflow is:

```text
Load CARS/BCARS spectrum
Preprocess / denoise signal
Estimate background
Apply background correction
Run phase retrieval
Apply optional phase correction
Inspect Raman-like output
Compare against reference or synthetic target
```

---

## Repository structure

```text
prCARS/
  examples/
    Example scripts and usage demos

  prcars/
    __init__.py
    pipeline.py
    result.py

    methods/
      kk.py
      mem.py
      nn.py

    corrections/
      background.py
      denoise.py
      phase.py
      phase_matching.py

    networks/
      Optional neural-network model utilities

    utils/
      Synthetic data, benchmarking, and plotting helpers

  tests/
    Unit and pipeline tests

  sanity_check.py
    Small local check script

  pyproject.toml
    Package metadata and dependencies
```

---

## Relationship to the CARS ecosystem

prCARS is designed to work as the retrieval layer in a broader CARS/BCARS workflow:

```text
CARSBench  → generate synthetic benchmark spectra
prCARS     → recover Raman-like spectra
CARSGuard  → validate plausibility and Raman consistency
```

Together, these projects form a small research ecosystem for simulation, retrieval, and validation of CARS/BCARS spectra.

---

## Limitations

prCARS is a research and educational software project.

Current limitations include:

* The package is not a substitute for experimental validation.
* Retrieval quality depends on preprocessing choices and input signal quality.
* Neural-network retrieval is experimental and optional.
* Real-data validation workflows are still planned.
* Results should be interpreted carefully when applied to real biological or clinical spectra.

This project is **not intended for clinical diagnosis, medical decision-making, or deployment in real healthcare settings**.

---

## Documentation

Additional documentation is available in the [`docs/`](docs/) folder.

Recommended pages:

- [`docs/methods.md`](docs/methods.md)
- [`docs/preprocessing.md`](docs/preprocessing.md)
- [`docs/examples.md`](docs/examples.md)

---

## Roadmap

Planned improvements include:

* Improve test coverage for retrieval methods and correction utilities
* Expand CI with optional backend tests for PyTorch or TensorFlow
* Add documentation pages for retrieval methods and preprocessing choices
* Add example figures to the README
* Add stronger synthetic benchmark reports
* Add integration examples with CARSBench
* Add validation examples with CARSGuard
* Add real CARS/BCARS data examples where licensing allows
* Improve neural-network retrieval documentation
* Add release notes and citation metadata

---

## Citation

If you use prCARS in research, education, or benchmarking work, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

```bibtex
@misc{prcars2026,
  title={prCARS: Phase Retrieval and Raman-like Signal Reconstruction for CARS/BCARS Spectroscopy},
  author={Houhou, Rola},
  year={2026},
  note={Alpha research software},
  url={https://github.com/rhouhou/prCARS}
}
```

---

## License

This project is licensed under the MIT License.
