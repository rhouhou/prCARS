# Integration with CARSBench and CARSGuard

prCARS is designed to work as the retrieval layer in a broader CARS/BCARS spectroscopy workflow.

The three related projects are:

| Project     | Role                                                                     |
| ----------- | ------------------------------------------------------------------------ |
| `CARSBench` | Simulates CARS/BCARS spectra under controlled domain shifts              |
| `prCARS`    | Retrieves Raman-like spectra from CARS/BCARS input                       |
| `CARSGuard` | Validates whether spectra and retrieval outputs are physically plausible |

Together, they form a small research ecosystem for simulation, retrieval, and validation.

---

## Ecosystem workflow

A typical workflow is:

```text
CARSBench simulated spectrum
        ↓
prCARS retrieval
        ↓
CARSGuard validation
        ↓
Benchmark metrics and QC reports
```

---

## Using prCARS with CARSBench

CARSBench can generate synthetic CARS/BCARS spectra and Raman-equivalent targets.

prCARS can then be used to retrieve Raman-like signals from the generated spectra.

Example workflow:

```python
import prcars as ca

# Example arrays loaded from a CARSBench generated sample
wavenumbers = sample.axis
cars_spectrum = sample.spectrum
raman_target = sample.raman_target

result = ca.retrieve(
    wavenumbers,
    cars_spectrum,
    method="kk",
    background="rolling_ball",
    correction="divide",
    denoise="savgol",
)

retrieved = result.im_chi3
```

The retrieved signal can then be compared with the CARSBench Raman-equivalent target.

Possible metrics include:

* RMSE
* MAE
* spectral angle
* Pearson correlation
* domain-level performance gap

---

## Cross-domain benchmark use

CARSBench defines domains such as:

```text
A_typical
B_high_res
C_low_res_noisy
D_calibration_shift
E_window_shift
F_nrb_family_shift
G_biochemical_source
H_biochemical_target
```

A useful benchmark experiment is:

```text
Generate spectra from each CARSBench domain
Run the same prCARS pipeline on every domain
Compare retrieval quality across domains
Identify where retrieval fails or degrades
```

This helps answer questions such as:

* Does the retrieval method fail under stronger noise?
* Does NRB-family shift reduce retrieval quality?
* Does calibration shift affect Raman-like recovery?
* Does biochemical composition shift create a performance gap?

---

## Using prCARS with CARSGuard

CARSGuard can be used after retrieval to check whether the input spectra and recovered Raman-like outputs are plausible.

Example workflow:

```text
Measured or simulated CARS spectrum
        ↓
prCARS retrieval
        ↓
Recovered Raman-like signal
        ↓
CARSGuard plausibility and consistency checks
```

CARSGuard can help evaluate:

* whether the CARS/BCARS spectrum looks physically plausible
* whether the retrieved Raman-like signal has suspicious artifacts
* whether the recovered spectrum agrees with reference Raman-like patterns
* whether a retrieval result should be trusted or inspected manually

---

## Recommended combined workflow

For a complete experiment:

```text
1. Generate synthetic spectra with CARSBench
2. Run prCARS retrieval on each spectrum
3. Compare retrieved signals with known Raman targets
4. Run CARSGuard validation on spectra and retrievals
5. Summarize results by domain and seed
6. Report where retrieval methods succeed or fail
```

---

## Example result table

A combined benchmark could produce a table like:

| Method            | Train/source domain    | Test domain            |  RMSE | Spectral angle | CARSGuard status |
| ----------------- | ---------------------- | ---------------------- | ----: | -------------: | ---------------- |
| KK + rolling-ball | `A_typical`            | `C_low_res_noisy`      | 0.084 |           0.19 | pass             |
| KK + ALS          | `A_typical`            | `F_nrb_family_shift`   | 0.112 |           0.27 | inspect          |
| MEM + SNIP        | `G_biochemical_source` | `H_biochemical_target` | 0.098 |           0.22 | pass             |

This type of table makes retrieval quality and validation status easier to interpret together.

---

## Notes

The integration workflow is currently a planned direction rather than a fully packaged end-to-end pipeline.

At this stage:

* `CARSBench` provides simulation and benchmark data generation.
* `prCARS` provides retrieval and preprocessing tools.
* `CARSGuard` provides validation and plausibility-checking tools.

Future work may include shared examples, common data loaders, and an end-to-end pipeline that connects all three projects.
