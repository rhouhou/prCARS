# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

## [0.1.0] – 2025-XX-XX

### Added
- `KramersKronig` retriever: FFT-based and direct principal-value Hilbert transform.
- `MaximumEntropy` retriever: Burg lattice and Yule-Walker AR solvers with KK or native phase retrieval.
- `NeuralNetRetriever`: unified PyTorch / TensorFlow backend, pretrained model registry, fine-tuning API.
- `Pipeline` orchestrator with pluggable background, correction, denoise, and retrieval steps.
- `retrieve()` top-level convenience function.
- Background estimators: ALS, polynomial, SNIP, rolling-ball.
- Denoising: Savitzky-Golay, Wiener, wavelet (PyWavelets).
- Phase-matching sinc² correction.
- Automatic constant-phase correction via silent-region optimisation.
- `CARSResult` dataclass with `save()`, `load()`, `plot()`, `peak_positions()`, `normalise()`.
- `utils` module: synthetic CARS generator, benchmark, compare_plot.
- `networks` module: weight registry and `download_weights()`.
- Full test suite (pytest) with parametrised coverage of all methods and corrections.
- GitHub Actions CI matrix (Ubuntu, macOS, Windows × Python 3.10–3.12).
