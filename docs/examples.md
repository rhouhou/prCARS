# Examples and benchmark workflow

prCARS includes synthetic examples and lightweight benchmark utilities for testing CARS/BCARS Raman-like retrieval workflows.

These examples are useful for checking that the retrieval pipeline works before applying it to experimental spectra.

---

## Synthetic CARS example

prCARS provides a synthetic CARS/BCARS-like example generator.

```python
from prcars.utils import synthetic_cars

wavenumbers, cars_raw, im_true = synthetic_cars(seed=0)
```

This returns:

| Variable      | Description                                  |
| ------------- | -------------------------------------------- |
| `wavenumbers` | Raman-shift axis                             |
| `cars_raw`    | Synthetic CARS/BCARS-like intensity spectrum |
| `im_true`     | Synthetic Raman-like target signal           |

The synthetic target is useful for checking whether a retrieval method is producing a reasonable Raman-like output.

---

## Basic Kramers-Kronig retrieval example

```python
import prcars as ca
from prcars.utils import synthetic_cars

wavenumbers, cars_raw, im_true = synthetic_cars(seed=0)

result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
)

print(result.im_chi3.shape)
```

The retrieved Raman-like signal is available as:

```python
result.im_chi3
```

---

## Retrieval with preprocessing

A more realistic example includes background estimation, correction, and denoising.

```python
import prcars as ca
from prcars.utils import synthetic_cars

wavenumbers, cars_raw, im_true = synthetic_cars(seed=0)

result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
    background="rolling_ball",
    correction="divide",
    denoise="savgol",
)
```

This workflow applies:

1. Savitzky-Golay denoising
2. rolling-ball background estimation
3. divide correction
4. Kramers-Kronig retrieval

---

## MEM retrieval example

```python
import prcars as ca
from prcars.utils import synthetic_cars

wavenumbers, cars_raw, im_true = synthetic_cars(seed=0)

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

MEM retrieval can be used to compare alternative retrieval assumptions against Kramers-Kronig retrieval.

---

## Pipeline example

The `Pipeline` object is useful when the same retrieval configuration should be applied to multiple spectra.

```python
import prcars as ca
from prcars.utils import synthetic_cars

pipeline = ca.Pipeline(
    method="kk",
    background="rolling_ball",
    correction="divide",
    denoise="savgol",
)

wavenumbers, cars_raw, im_true = synthetic_cars(seed=0)

result = pipeline.run(wavenumbers, cars_raw)
```

This makes retrieval settings explicit and reusable.

---

## Benchmarking retrieval methods

prCARS includes lightweight benchmark utilities for comparing retrieval outputs against a known synthetic target.

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

The benchmark output can be used to compare retrieval quality across methods.

Possible metrics may include:

* correlation with the synthetic target
* RMSE
* MAE
* shape similarity

The exact available metrics depend on the utility implementation.

---

## Plotting results

If plotting dependencies are installed, retrieval results can be visualized with standard Python tools.

```python
import matplotlib.pyplot as plt
import prcars as ca
from prcars.utils import synthetic_cars

wavenumbers, cars_raw, im_true = synthetic_cars(seed=0)

result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
)

plt.plot(wavenumbers, im_true, label="Synthetic target")
plt.plot(wavenumbers, result.im_chi3, label="Retrieved")
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Signal")
plt.legend()
plt.show()
```

Install plotting support with:

```bash
python -m pip install -e ".[plot]"
```

On some systems, use `python3` instead of `python`.

---

## Recommended example workflow

A practical development workflow is:

```text
Generate synthetic CARS example
Run KK retrieval without preprocessing
Run KK retrieval with preprocessing
Run MEM retrieval
Compare outputs against synthetic target
Inspect plots
Adjust preprocessing choices
Document selected settings
```

This helps identify whether a retrieval failure comes from the retrieval method itself, the preprocessing configuration, or the synthetic input conditions.

---

## Relationship to CARSBench

Synthetic examples in prCARS are useful for quick tests.

For larger cross-domain experiments, use CARSBench to generate benchmark spectra and then apply prCARS retrieval pipelines to each domain.

Example ecosystem workflow:

```text
CARSBench synthetic spectrum
        ↓
prCARS retrieval
        ↓
CARSGuard validation
```

---

## Notes

The examples in this document are intended for small-scale testing and demonstration.

For scientific reporting, always document:

* retrieval method
* preprocessing steps
* background method
* correction mode
* denoising method
* random seed
* whether data are synthetic or experimental
* evaluation metric definitions

prCARS is intended for research, education, benchmarking, and portfolio demonstration. It is not intended for clinical diagnosis or deployment.
