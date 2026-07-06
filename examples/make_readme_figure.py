from pathlib import Path

import matplotlib.pyplot as plt

import prcars as ca
from prcars.utils import synthetic_cars


def normalize(signal):
    signal = signal.copy()
    signal = signal - signal.min()
    max_value = signal.max()
    if max_value > 0:
        signal = signal / max_value
    return signal


def main():
    output_dir = Path("docs/assets")
    output_dir.mkdir(parents=True, exist_ok=True)

    wavenumbers, cars_raw, im_true = synthetic_cars(seed=0)

    result = ca.retrieve(
        wavenumbers,
        cars_raw,
        method="kk",
        background="rolling_ball",
        correction="divide",
        denoise="savgol",
    )

    plt.figure(figsize=(8, 4))
    plt.plot(wavenumbers, normalize(cars_raw), label="Synthetic CARS/BCARS input")
    plt.plot(wavenumbers, normalize(im_true), label="Synthetic Raman-like target")
    plt.plot(wavenumbers, normalize(result.im_chi3), label="prCARS retrieved signal")

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Normalized signal")
    plt.title("prCARS synthetic retrieval example")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "example_retrieval.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()