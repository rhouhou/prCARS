from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import prcars as ca
import os
from dotenv import load_dotenv

load_dotenv()
data_dir = Path(os.environ["DATA_DIR"])

# Find all .npz files
files = sorted(data_dir.glob("*.npz"))
if not files:
    raise FileNotFoundError(f"No .npz files found in {data_dir}")

# Pick one random batch file
batch_file = random.choice(files)
print("Using batch file:", batch_file)

# Load the batch
data = np.load(batch_file)
print("Keys in file:", data.files)


# Adjust these names to match your file
wn = data["axis"]          # shape: (L,)
cars_batch = data["spectrum"]         # shape: (N, L)

# Optional truth, if present
im_true_batch = data["raman_target"] if "raman_target" in data.files else None

# Pick one random spectrum from the batch
idx = np.random.randint(len(cars_batch))
wn = wn[idx]
cars = cars_batch[idx]
im_true = im_true_batch[idx] if im_true_batch is not None else None

print("Random spectrum index:", idx)
print("cars shape:", cars.shape)

# Run your code
kk = ca.KramersKronig()
res_raw = kk.retrieve(wn, cars)

res_pipe = ca.retrieve(
    wn,
    cars,
    method="kk",
    background="none",
    denoise="savgol",
    auto_phase=False,
)

# Normalize for shape comparison
def norm(y):
    y = np.asarray(y, dtype=float)
    return y / (np.max(np.abs(y)) + 1e-30)

plt.figure(figsize=(8, 5))
plt.plot(wn, cars, label="cars")
plt.plot(wn, im_true, label="true Im(χ3)")
plt.plot(wn, res_raw["im_chi3"], label="raw KK")
#plt.plot(wn, res_pipe.im_chi3, label="pipeline KK")
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Normalized signal")
plt.legend()
plt.tight_layout()
plt.show()