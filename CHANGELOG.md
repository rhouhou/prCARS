# Changelog

All notable changes to prCARS will be documented in this file.

This project follows a simple versioned changelog format.

---

## [0.1.0-alpha] - 2026-07-06

### Added

- Initial alpha version of prCARS
- Kramers-Kronig phase retrieval workflow
- Maximum Entropy Method retrieval workflow
- Optional neural-network retrieval interface
- Background estimation utilities:
  - ALS
  - polynomial fitting
  - SNIP
  - rolling-ball
- Background correction modes:
  - subtract
  - divide
  - square-root divide
- Denoising utilities:
  - Savitzky-Golay
  - Wiener
  - wavelet-based denoising
- Synthetic CARS/BCARS example generation
- Benchmark utilities for comparing retrieval outputs
- Pipeline object for reproducible retrieval workflows
- Basic tests for retrieval, preprocessing, and pipeline behavior
- GitHub Actions CI for linting and tests
- Citation metadata with `CITATION.cff`
- Improved README with installation, quickstart, methods, limitations, and roadmap

### Fixed

- Fixed Kramers-Kronig direct phase retrieval method call
- Updated optional neural-network tests to skip when PyTorch or TensorFlow is not installed
- Tuned Ruff configuration for the current alpha-stage scientific codebase
- Cleaned repository hygiene and ignored local system/generated files

### Notes

- Neural-network retrieval is experimental and requires optional backend dependencies.
- Generated data and large outputs are not included in the repository.
- This project is for research, education, and portfolio demonstration.
- It is not intended for clinical diagnosis, medical decision-making, or deployment in real healthcare settings.