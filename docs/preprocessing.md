# Preprocessing and correction

prCARS includes preprocessing, background estimation, correction, denoising, and optional phase-correction utilities for CARS/BCARS spectra.

These steps are important because CARS/BCARS spectra are affected by non-resonant background, noise, baseline effects, and instrument-related distortions before Raman-like information can be recovered.

---

## Typical preprocessing workflow

A typical prCARS workflow is:

```text
Load CARS/BCARS spectrum
Apply optional phase-matching correction
Apply optional denoising
Estimate background
Apply background correction
Run phase retrieval
Apply optional auto-phase correction
Inspect Raman-like output
```

In code, this is usually handled through the `Pipeline` interface:

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

result = pipeline.run(wavenumbers, cars_raw)
```

---

## Background estimation

Background estimation tries to approximate the smooth non-resonant or baseline-like contribution in the measured CARS/BCARS spectrum.

prCARS currently supports:

| Method         | Description                                     |
| -------------- | ----------------------------------------------- |
| `als`          | Asymmetric least-squares background estimation  |
| `polynomial`   | Polynomial background fitting                   |
| `snip`         | SNIP-style background estimation                |
| `rolling_ball` | Rolling-ball style smooth background estimation |
| `none`         | No background estimation                        |

---

## ALS background

ALS is useful when the background is smooth and the signal contains sharper spectral features.

Example:

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
    background="als",
    correction="divide",
)
```

Use ALS when:

* the background is smooth
* peaks are sharper than the baseline
* a flexible baseline estimate is needed

---

## Polynomial background

Polynomial fitting is a simple background model.

Example:

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
    background="polynomial",
    correction="subtract",
)
```

Use polynomial background when:

* the background is globally smooth
* a simple interpretable baseline is preferred
* the spectral window is not too complex

---

## SNIP background

SNIP-style background estimation can be useful for spectra with slowly varying background and peak-like structures.

Example:

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="mem",
    background="snip",
    correction="divide",
)
```

Use SNIP when:

* peak-like structures should be preserved
* the background is broad and smooth
* you want an alternative to ALS

---

## Rolling-ball background

Rolling-ball background estimation is useful for smooth baseline removal.

Example:

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
    background="rolling_ball",
    correction="divide",
)
```

Use rolling-ball background when:

* the background is smooth
* local baseline behavior matters
* you want a robust non-parametric background estimate

---

## Background correction modes

After estimating a background, prCARS can correct the measured spectrum before retrieval.

| Correction    | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| `subtract`    | Subtracts the estimated background from the signal                 |
| `divide`      | Divides the signal by the estimated background                     |
| `sqrt_divide` | Applies square-root division, useful for amplitude-like correction |
| `none`        | Does not apply correction                                          |

---

## Subtract correction

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
    background="polynomial",
    correction="subtract",
)
```

Use `subtract` when the background behaves like an additive baseline.

---

## Divide correction

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
    background="als",
    correction="divide",
)
```

Use `divide` when the background behaves more like a multiplicative or normalization factor.

This is often useful in CARS/BCARS workflows because the non-resonant contribution can affect the overall intensity envelope.

---

## Square-root divide correction

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
    background="als",
    correction="sqrt_divide",
)
```

Use `sqrt_divide` when working closer to an amplitude-domain correction.

---

## Denoising

prCARS includes optional denoising before retrieval.

| Method    | Description              |
| --------- | ------------------------ |
| `savgol`  | Savitzky-Golay smoothing |
| `wiener`  | Wiener filtering         |
| `wavelet` | Wavelet-based denoising  |
| `none`    | No denoising             |

---

## Savitzky-Golay denoising

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
    denoise="savgol",
)
```

Use Savitzky-Golay denoising when:

* noise is moderate
* peak shapes should be preserved
* a simple smoothing method is enough

---

## Wiener denoising

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
    denoise="wiener",
)
```

Use Wiener denoising when:

* the noise is approximately stationary
* adaptive smoothing may help
* a simple signal-processing baseline is needed

---

## Wavelet denoising

Wavelet denoising requires the optional wavelet dependency.

Install with:

```bash
python -m pip install -e ".[wavelet]"
```

Then use:

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
    denoise="wavelet",
)
```

Use wavelet denoising when:

* spectra contain multi-scale noise
* you want a stronger denoising option
* preserving local spectral structure is important

---

## Auto-phase correction

prCARS supports optional automatic phase correction using a silent spectral region.

Example:

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
    auto_phase=True,
    silent_region=(2700, 2730),
)
```

Use auto-phase correction when:

* a silent or low-signal region is known
* the retrieved Raman-like signal has a phase offset
* you want to reduce baseline-like phase artifacts

The selected silent region should be chosen carefully based on the spectral window and expected chemistry.

---

## Recommended starting configurations

### Simple KK baseline

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
    background="none",
    correction="none",
    denoise="none",
)
```

Use this first to check raw retrieval behavior.

---

### Practical KK workflow

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
    background="rolling_ball",
    correction="divide",
    denoise="savgol",
)
```

This is a good practical starting point for synthetic examples.

---

### Stronger preprocessing workflow

```python
result = ca.retrieve(
    wavenumbers,
    cars_raw,
    method="kk",
    background="als",
    correction="divide",
    denoise="wavelet",
    auto_phase=True,
    silent_region=(2700, 2730),
)
```

Use this when spectra are noisier or require stronger correction.

---

## Notes

Preprocessing choices can strongly affect retrieval quality.

For fair method comparison, always report:

* retrieval method
* background method
* correction mode
* denoising method
* phase-correction settings
* spectral window
* whether synthetic or real data were used

prCARS is intended for research, education, benchmarking, and portfolio demonstration. It is not intended for clinical diagnosis or deployment.
