# Retrieval methods

prCARS provides modular retrieval methods for recovering Raman-like spectral information from CARS/BCARS spectra.

The package currently supports:

* Kramers-Kronig retrieval
* Maximum Entropy Method retrieval
* optional neural-network retrieval

These methods can be used directly or through the high-level `Pipeline` interface.

---

## Kramers-Kronig retrieval

Kramers-Kronig retrieval is the main classical phase-retrieval method in prCARS.

It estimates the phase of the CARS field from the log-amplitude and reconstructs a Raman-like signal from the imaginary part of the recovered complex response.

### Basic usage

```python
import prcars as ca

result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
)
```

The output is a `CARSResult` object.

Important fields include:

| Field                | Description                          |
| -------------------- | ------------------------------------ |
| `result.wavenumbers` | Raman-shift axis                     |
| `result.amplitude`   | Retrieved amplitude                  |
| `result.phase`       | Retrieved phase                      |
| `result.im_chi3`     | Raman-like imaginary component       |
| `result.re_chi3`     | Real component                       |
| `result.background`  | Estimated background, when available |

### Advanced usage

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
    retriever_kw={
        "zero_pad_factor": 8,
    },
)
```

### When to use it

Kramers-Kronig retrieval is a good first choice when:

* the spectrum is reasonably smooth
* the non-resonant background can be estimated
* the signal is suitable for classical phase retrieval
* a fast and interpretable method is preferred

---

## Maximum Entropy Method retrieval

Maximum Entropy Method retrieval provides an alternative phase-retrieval approach based on autoregressive spectral modeling.

It can be useful for testing whether retrieval behavior is stable across different assumptions.

### Basic usage

```python
import prcars as ca

result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="mem",
)
```

### Advanced usage

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

### Available MEM options

| Option            | Meaning                                         |
| ----------------- | ----------------------------------------------- |
| `order`           | Autoregressive model order                      |
| `solver`          | Solver type, such as `burg` or `yulewalker`     |
| `phase_method`    | Phase extraction strategy                       |
| `zero_pad_factor` | Optional padding factor for spectral processing |

### When to use it

MEM retrieval is useful when:

* comparing alternative classical retrieval assumptions
* testing sensitivity to phase-retrieval method
* evaluating robustness on synthetic spectra
* benchmarking against Kramers-Kronig retrieval

---

## Optional neural-network retrieval

prCARS includes an optional neural-network retrieval interface.

This interface is experimental and requires an additional backend such as PyTorch or TensorFlow.

### Installation

For PyTorch support:

```bash
python -m pip install -e ".[torch]"
```

For TensorFlow support:

```bash
python -m pip install -e ".[tensorflow]"
```

### Usage

```python
import prcars as ca

result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="nn",
    retriever_kw={
        "model_name": "cars_unet_v1",
        "backend": "torch",
    },
)
```

### Current status

Neural-network retrieval is experimental.

It is included as a future-facing interface for learned CARS/BCARS retrieval workflows, but the classical methods should be treated as the main implemented methods at this stage.

---

## Recommended comparison workflow

For method comparison, use the same spectrum and preprocessing settings across retrieval methods.

Example:

```python
from prcars.utils import synthetic_cars

wavenumbers, cars_raw, im_true = synthetic_cars(seed=0)

kk_result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
)

mem_result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="mem",
)
```

Then compare:

* recovered Raman-like signal shape
* correlation with synthetic target
* RMSE or MAE
* stability under noise
* sensitivity to background correction
* behavior across different preprocessing options

---

## Notes

Retrieval quality depends strongly on:

* input signal quality
* background estimation
* denoising choice
* correction mode
* spectral window
* noise level
* whether the non-resonant background assumption is reasonable

prCARS is intended for research, education, benchmarking, and portfolio demonstration. It is not intended for clinical diagnosis or deployment.
