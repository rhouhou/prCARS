# prcars

> **CARS spectral retrieval in Python** — Kramers-Kronig · Maximum Entropy · Neural Networks

[![PyPI](https://img.shields.io/pypi/v/prcars)](https://pypi.org/project/prcars/)
[![Python](https://img.shields.io/pypi/pyversions/prcars)](https://pypi.org/project/prcars/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/rhouhou/prcars/actions/workflows/ci.yml/badge.svg)](https://github.com/rhouhou/prcars/actions)

`prcars` extracts the imaginary part of the third-order susceptibility
Im[χ⁽³⁾] from raw Coherent Anti-Stokes Raman Scattering (CARS) spectra.
The package provides a **unified pipeline** where you choose:

| Step | Options |
|------|---------|
| Phase-matching correction | sinc² envelope |
| Denoising | Savitzky-Golay · Wiener · Wavelet |
| Background estimation | ALS · Polynomial · SNIP · Rolling-ball |
| Background correction | Subtract · Divide · √-Divide |
| **Retrieval** | **KK · MEM · Neural Network (pretrained or fine-tuned)** |
| Phase correction | Automatic (silent-region optimisation) |

---

## Installation

```bash
# core (KK + MEM)
pip install prcars

# with wavelet denoising
pip install "prcars[wavelet]"

# with plotting helpers
pip install "prcars[plot]"

# with PyTorch neural-network support
pip install "prcars[torch]"

# everything
pip install "prcars[wavelet,plot]"
```

---

## Quick start

```python
import numpy as np
import prcars as ca

# load your data
wavenumbers = np.load("wavenumbers.npy")   # cm⁻¹
cars_raw    = np.load("cars_spectrum.npy")

# one-liner with defaults  (KK + ALS background + Savitzky-Golay)
result = ca.retrieve(wavenumbers, cars_raw)

print(result.im_chi3)          # Im[χ³] – your Raman-like spectrum
print(result.peak_positions())  # wavenumbers of detected peaks
result.plot()                   # diagnostic 3-panel figure
result.save("result.npz")       # save to disk
```

---

## Choosing the retrieval method

### Kramers-Kronig (default)

```python
result = ca.retrieve(wavenumbers, cars_raw, method="kk")

# advanced KK options
result = ca.retrieve(
    wavenumbers, cars_raw,
    method       = "kk",
    background   = "als",          # 'als' | 'polynomial' | 'snip' | 'rolling_ball'
    correction   = "divide",       # 'divide' | 'subtract' | 'sqrt_divide'
    denoise      = "savgol",       # 'savgol' | 'wiener' | 'wavelet' | 'none'
    auto_phase   = True,
    silent_region= (2700, 2730),   # cm⁻¹ window with no Raman peaks
    retriever_kw = {"zero_pad_factor": 8},
)
```

### Maximum Entropy Method

```python
result = ca.retrieve(
    wavenumbers, cars_raw,
    method       = "mem",
    background   = "snip",
    retriever_kw = {
        "order"        : 128,          # AR model order (None → auto)
        "solver"       : "burg",       # 'burg' | 'yulewalker'
        "phase_method" : "kk",         # 'kk' | 'mem_phase'
    },
)
```

### Neural Network (pretrained)

```python
# download bundled weights once
import prcars.networks as nets
nets.download_weights()

result = ca.retrieve(
    wavenumbers, cars_raw,
    method       = "nn",
    background   = "als",
    retriever_kw = {
        "model_name" : "cars_unet_v1",   # or 'cars_cnn_v1'
        "backend"    : "torch",           # 'torch' | 'tensorflow' | None (auto)
    },
)
```

### Neural Network – fine-tune on your own data

```python
from prcars import NeuralNetRetriever

nn = NeuralNetRetriever(model_name="cars_unet_v1")

# X_train: (N, L) array of CARS spectra
# y_train: (N, L) array of ground-truth Im[χ³]
nn.fine_tune(X_train, y_train, epochs=50, lr=5e-5)
nn.save("my_model.pt")

# use the fine-tuned model
result = ca.retrieve(wavenumbers, cars_raw,
                     method="nn",
                     retriever_kw={"model_path": "my_model.pt"})
```

---

## Pipeline object

For reproducible, reusable workflows use `Pipeline` directly:

```python
pipeline = ca.Pipeline(
    method        = "mem",
    background    = "als",
    correction    = "divide",
    denoise       = "wavelet",
    denoise_kw    = {"wavelet": "sym5", "level": 4},
    background_kw = {"lam": 1e6, "p": 0.005},
    auto_phase    = True,
    silent_region = (2700, 2730),
)

result = pipeline.run(wavenumbers, spectrum_1)
result2 = pipeline.run(wavenumbers, spectrum_2)   # same settings
```

---

## Access intermediate results

```python
result = pipeline.run(wavenumbers, cars_raw)

# intermediate arrays stored in result.intermediate
bg_corrected = result.intermediate["after_correction"]
denoised     = result.intermediate["after_denoise"]
```

---

## Benchmarking against ground truth

```python
from prcars.utils import synthetic_cars, benchmark, compare_plot

wn, cars, im_true = synthetic_cars()

scores = benchmark(wn, cars, im_true, methods=["kk", "mem"])
for method, data in scores.items():
    print(f"{method}: MSE={data['mse']:.5f}  r={data['pearson']:.4f}")

compare_plot(wn, im_true, scores)
```

---

## Background methods at a glance

| Method | Best for | Key parameters |
|--------|----------|---------------|
| `als` | Smoothly varying background | `lam` (smoothness), `p` (asymmetry) |
| `polynomial` | Known signal-free regions | `degree`, `mask` |
| `snip` | Noisy spectra, no manual tuning | `max_iterations` |
| `rolling_ball` | Broad undulating background | `radius` |

---

## Phase-matching correction

For loosely focused or non-collinear geometries pass the wave-vector mismatch:

```python
result = ca.retrieve(
    wavenumbers, cars_raw,
    phase_matching = {"delta_k": 0.02, "interaction_length": 0.3},
)
```

Set `phase_matching=None` (default) to skip (appropriate for tight focusing).

---

## Adding your own pretrained model

Drop a `.pt` (PyTorch) or `.h5` (Keras) file into `~/.prcars/models/`
and pass its stem as `model_name`:

```python
# file: ~/.prcars/models/my_awesome_model.pt
result = ca.retrieve(wn, cars,
                     method="nn",
                     retriever_kw={"model_name": "my_awesome_model"})
```

---

## Project structure

```
prcars/
├── __init__.py          # public API
├── pipeline.py          # Pipeline orchestrator + retrieve()
├── result.py            # CARSResult dataclass
├── methods/
│   ├── kk.py            # Kramers-Kronig
│   ├── mem.py           # Maximum Entropy Method
│   └── nn.py            # Neural Network retriever
├── corrections/
│   ├── background.py    # estimators + correction
│   ├── denoise.py       # Savitzky-Golay, Wiener, Wavelet
│   ├── phase.py         # auto_phase_correction
│   └── phase_matching.py
├── networks/
│   └── __init__.py      # weight registry + download_weights()
└── utils/
    └── __init__.py      # synthetic data, benchmark, compare_plot
tests/
examples/
docs/
```

---

## Contributing

1. Fork the repo and create a feature branch.
2. Install dev dependencies: `pip install -e ".[dev]"`
3. Run tests: `pytest`
4. Lint: `ruff check .`
5. Open a pull request.

---

## Citation

If you use this package in published work, please cite:

```bibtex
@software{prcars,
  author  = {Dr. Rola Houhou},
  title   = {prcars: CARS spectral retrieval in Python},
  year    = {2026},
  url     = {https://github.com/rhouhou/prcars},
}
```

Key references implemented here:
- E.M. Vartiainen et al., *J. Appl. Phys.* **75**, 2815 (1994) — MEM
- Y. Liu et al., *Opt. Lett.* **34**, 1363 (2009) — MEM-CARS
- C.H. Camp Jr. & M.T. Cicerone, *Nat. Photon.* **9**, 295 (2015) — KK review
- P.H.C. Eilers & H.F.M. Boelens (2005) — ALS baseline

---

## License

MIT © Dr. Rola Houhou
