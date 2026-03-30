"""
prcars.networks
----------------------
Pretrained model weight management.

Weights are NOT shipped with the package (they are too large for PyPI).
Use :func:`download_weights` to fetch them from the project's GitHub release.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

_PKG_DIR = Path(__file__).parent

# name → (filename, sha256, download_url)
_REGISTRY: dict[str, tuple[str, str, str]] = {
    "cars_unet_v1": (
        "cars_unet_v1.pt",
        "PLACEHOLDER_SHA256",                          # fill after training
        "https://github.com/your-org/cars-analysis"
        "/releases/download/v0.1.0/cars_unet_v1.pt",
    ),
    "cars_cnn_v1": (
        "cars_cnn_v1.pt",
        "PLACEHOLDER_SHA256",
        "https://github.com/your-org/cars-analysis"
        "/releases/download/v0.1.0/cars_cnn_v1.pt",
    ),
}


def download_weights(
    model_name: str | None = None,
    *,
    force: bool = False,
    dest_dir: Path | str | None = None,
) -> None:
    """
    Download pretrained model weights.

    Parameters
    ----------
    model_name : str or None
        Name of the model to download.  If ``None``, all models are downloaded.
    force : bool
        Re-download even if the file already exists. Default ``False``.
    dest_dir : path-like, optional
        Directory to save weights. Defaults to the ``networks/`` folder inside
        the installed package.

    Examples
    --------
    >>> import prcars.networks as nets
    >>> nets.download_weights()                       # download all
    >>> nets.download_weights('cars_unet_v1')         # single model
    """
    dest = Path(dest_dir) if dest_dir else _PKG_DIR
    dest.mkdir(parents=True, exist_ok=True)

    targets = [model_name] if model_name else list(_REGISTRY.keys())

    for name in targets:
        if name not in _REGISTRY:
            raise ValueError(
                f"Unknown model '{name}'. Available: {list(_REGISTRY.keys())}"
            )
        filename, expected_sha, url = _REGISTRY[name]
        out_path = dest / filename

        if out_path.exists() and not force:
            print(f"[prcars] '{name}' already downloaded at {out_path}")
            continue

        print(f"[prcars] Downloading '{name}' from {url} …")
        try:
            urllib.request.urlretrieve(url, out_path)
        except Exception as e:
            print(f"[prcars] Download failed: {e}")
            continue

        # Verify checksum (skip if placeholder)
        if expected_sha != "PLACEHOLDER_SHA256":
            sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
            if sha != expected_sha:
                out_path.unlink()
                raise RuntimeError(
                    f"Checksum mismatch for '{name}'!\n"
                    f"  expected : {expected_sha}\n"
                    f"  got      : {sha}\n"
                    "The file has been removed. Try downloading again."
                )
        print(f"[prcars] '{name}' saved to {out_path}")


def list_available() -> list[str]:
    """List the names of all registered pretrained models."""
    return list(_REGISTRY.keys())
