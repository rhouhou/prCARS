"""
prcars.methods.nn
------------------------
Neural-network-based CARS retrieval.

Three usage modes
-----------------
1. **Pretrained** – load a bundled or user-supplied model by name/path.
2. **Fine-tune**  – continue training a pretrained model on new data.
3. **Train from scratch** – supply your own architecture and training data.

Bundled models
--------------
``cars_unet_v1``
    1-D U-Net trained on synthetic broadband CARS spectra (CH-stretch region,
    2700–3100 cm⁻¹).  Predicts Im[χ³] directly from raw intensity.

``cars_cnn_v1``
    Lightweight 1-D CNN; faster than U-Net, slightly lower accuracy.

Adding your own pretrained weights
-----------------------------------
Drop a ``.pt`` (PyTorch) or ``.h5`` / ``.keras`` (TensorFlow/Keras) file in
``~/.prcars/models/`` and pass ``model_name=<filename_without_ext>``.

Dependencies
------------
This module requires *either* ``torch`` or ``tensorflow`` (and optionally
``keras``).  If neither is available the class raises ``ImportError`` on
instantiation with a helpful message.
"""
from __future__ import annotations

import os
import warnings
import importlib
from pathlib import Path
from typing import Optional, Union
import numpy as np


# ── model registry ────────────────────────────────────────────────────────────

_BUNDLED_MODELS: dict[str, str] = {
    # name  →  relative path inside the package (prcars/networks/)
    "cars_unet_v1": "cars_unet_v1.pt",
    "cars_cnn_v1":  "cars_cnn_v1.pt",
}

_USER_MODEL_DIR = Path.home() / ".prcars" / "models"


def list_models() -> list[str]:
    """Return names of all available pretrained models."""
    names = list(_BUNDLED_MODELS.keys())
    if _USER_MODEL_DIR.exists():
        for p in _USER_MODEL_DIR.iterdir():
            if p.suffix in (".pt", ".h5", ".keras"):
                names.append(p.stem)
    return sorted(set(names))


# ── backend detection ─────────────────────────────────────────────────────────

def _detect_backend() -> str:
    for name in ("torch", "tensorflow"):
        if importlib.util.find_spec(name) is not None:
            return name
    raise ImportError(
        "prcars neural-network retrieval requires PyTorch or TensorFlow.\n"
        "Install one of:\n"
        "    pip install torch\n"
        "    pip install tensorflow\n"
    )


# ── PyTorch architecture ──────────────────────────────────────────────────────

def _build_unet_torch(in_channels: int = 1, depth: int = 4):
    """1-D U-Net implemented in PyTorch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvBlock(nn.Module):
        def __init__(self, ch_in, ch_out):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(ch_in, ch_out, 3, padding=1),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(inplace=True),
                nn.Conv1d(ch_out, ch_out, 3, padding=1),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class UNet1D(nn.Module):
        def __init__(self, depth=4, base_ch=32):
            super().__init__()
            chs = [base_ch * 2 ** i for i in range(depth)]   # [32, 64, 128, 256]

            # Encoder
            self.enc = nn.ModuleList(
                [ConvBlock(in_channels if i == 0 else chs[i - 1], chs[i])
                 for i in range(depth)]
            )
            self.pool = nn.ModuleList([nn.MaxPool1d(2) for _ in range(depth - 1)])

            # Bottleneck
            self.bot = ConvBlock(chs[-1], chs[-1] * 2)       # 256 -> 512

            # Decoder uses only the stored skip connections from enc[:-1]
            skip_chs = chs[:-1][::-1]                        # [128, 64, 32]
            up_in_chs = [chs[-1] * 2] + [chs[-1 - i] for i in range(1, depth - 1)]
            # for depth=4: [512, 256, 128]
            up_out_chs = skip_chs                            # [128, 64, 32]

            self.up = nn.ModuleList([
                nn.ConvTranspose1d(ch_in, ch_out, 2, stride=2)
                for ch_in, ch_out in zip(up_in_chs, up_out_chs)
            ])

            self.dec = nn.ModuleList([
                ConvBlock(ch_out + skip, ch_out)
                for ch_out, skip in zip(up_out_chs, skip_chs)
            ])
            # For depth=4:
            # up:  512->128, 256->64, 128->32
            # dec: (128+128)->128, (64+64)->64, (32+32)->32

            self.head = nn.Conv1d(chs[0], 1, 1)

        def forward(self, x):
            skips = []
            for enc, pool in zip(self.enc[:-1], self.pool):
                x = enc(x)
                skips.append(x)
                x = pool(x)

            x = self.enc[-1](x)
            x = self.bot(x)

            for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
                x = up(x)
                if x.shape[-1] != skip.shape[-1]:
                    x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
                x = torch.cat([skip, x], dim=1)
                x = dec(x)

            return self.head(x)

    return UNet1D(depth=depth)

def _build_cnn_torch():
    """Lightweight 1-D CNN (encoder-decoder style)."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv1d(1, 32, 9, padding=4), nn.ReLU(),
        nn.Conv1d(32, 64, 7, padding=3), nn.ReLU(),
        nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
        nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(),
        nn.Conv1d(32, 1, 1),
    )


# ── TensorFlow / Keras architecture ──────────────────────────────────────────

def _build_unet_keras(input_len: int = 512, depth: int = 4):
    """1-D U-Net implemented in Keras."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    def conv_block(x, filters):
        x = layers.Conv1D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        return x

    inp = keras.Input(shape=(input_len, 1))
    x   = inp
    chs = [32 * 2 ** i for i in range(depth)]
    skips = []
    for ch in chs[:-1]:
        x = conv_block(x, ch)
        skips.append(x)
        x = layers.MaxPool1D(2)(x)
    x = conv_block(x, chs[-1])
    for ch, skip in zip(reversed(chs[:-1]), reversed(skips)):
        x = layers.UpSampling1D(2)(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, ch)
    out = layers.Conv1D(1, 1, activation="linear")(x)
    return keras.Model(inp, out)


# ── main class ────────────────────────────────────────────────────────────────

class NeuralNetRetriever:
    """
    Neural-network-based CARS spectral retrieval.

    Parameters
    ----------
    model_name : str or None
        Name of a bundled pretrained model (``'cars_unet_v1'``,
        ``'cars_cnn_v1'``) or the stem of a file in
        ``~/.prcars/models/``.  If ``None`` a fresh U-Net is built
        without pretrained weights.
    model_path : str or Path, optional
        Explicit path to a ``.pt`` or ``.h5`` file; overrides *model_name*.
    architecture : str
        Which architecture to instantiate when no weights are loaded.
        One of ``'unet'``, ``'cnn'``.  Default ``'unet'``.
    backend : str or None
        ``'torch'`` or ``'tensorflow'``.  If ``None``, auto-detected.
    input_len : int
        Expected spectrum length.  Used to build the Keras model.
    device : str
        PyTorch device string (``'cpu'``, ``'cuda'``, ``'mps'``).
    normalize_input : bool
        Min-max normalise the input before inference. Default ``True``.

    Examples
    --------
    >>> from prcars.methods.nn import NeuralNetRetriever
    >>> nn = NeuralNetRetriever(model_name='cars_unet_v1')
    >>> result_dict = nn.retrieve(wavenumbers, corrected_intensity)

    Training / fine-tuning
    ----------------------
    >>> nn = NeuralNetRetriever(model_name='cars_unet_v1')
    >>> nn.fine_tune(X_train, y_train, epochs=20, lr=1e-4)
    >>> nn.save('my_finetuned_model.pt')
    """

    def __init__(
        self,
        *,
        model_name: str | None = "cars_unet_v1",
        model_path: str | Path | None = None,
        architecture: str = "unet",
        backend: str | None = None,
        input_len: int = 512,
        device: str = "cpu",
        normalize_input: bool = True,
    ):
        self.model_name      = model_name
        self.architecture    = architecture
        self.input_len       = input_len
        self.device_str      = device
        self.normalize_input = normalize_input

        self.backend = backend or _detect_backend()
        self._model  = None

        resolved_path = model_path or self._resolve_model_path(model_name)
        self._load_or_build(resolved_path)

    # ── model loading ─────────────────────────────────────────────────────────
    def _resolve_model_path(self, name: str | None) -> Path | None:
        if name is None:
            return None
        # bundled
        if name in _BUNDLED_MODELS:
            pkg_dir = Path(__file__).parent.parent / "networks"
            p = pkg_dir / _BUNDLED_MODELS[name]
            if p.exists():
                return p
            warnings.warn(
                f"Bundled model weights '{name}' not found at {p}.\n"
                "The model will be initialised with random weights.\n"
                "Download pretrained weights with:\n"
                "    prcars.networks.download_weights()\n"
                "or visit: https://github.com/your-org/cars-analysis/releases",
                stacklevel=3,
            )
            return None
        # user models
        p = _USER_MODEL_DIR / (name + ".pt")
        if p.exists():
            return p
        p = _USER_MODEL_DIR / (name + ".h5")
        if p.exists():
            return p
        warnings.warn(
            f"Model '{name}' not found in {_USER_MODEL_DIR}. "
            "Using randomly initialised weights.", stacklevel=3
        )
        return None

    def _load_or_build(self, path: Path | None):
        if self.backend == "torch":
            self._load_torch(path)
        else:
            self._load_keras(path)

    def _load_torch(self, path: Path | None):
        import torch
        self._torch = torch
        self._device = torch.device(self.device_str)

        arch = self.architecture.lower()
        if arch == "unet":
            model = _build_unet_torch()
        elif arch == "cnn":
            model = _build_cnn_torch()
        else:
            raise ValueError(f"Unknown architecture '{arch}'")

        if path is not None and path.exists():
            state = torch.load(path, map_location=self._device)
            # support both raw state-dict and full checkpoint
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            try:
                model.load_state_dict(state)
            except RuntimeError as e:
                warnings.warn(f"Could not load weights from {path}: {e}")
        model.to(self._device)
        model.eval()
        self._model = model

    def _load_keras(self, path: Path | None):
        import tensorflow as tf
        self._tf = tf

        if path is not None and path.exists():
            self._model = tf.keras.models.load_model(str(path))
        else:
            self._model = _build_unet_keras(input_len=self.input_len)

    # ── inference ─────────────────────────────────────────────────────────────
    def retrieve(
        self,
        wavenumbers: np.ndarray,
        intensity: np.ndarray,
        *,
        nr_background: np.ndarray | None = None,
    ) -> dict:
        """
        Run neural-network inference and return the retrieval dict.

        The network predicts Im[χ³] directly from the (pre-processed) input
        spectrum.  Re[χ³] is estimated via a KK transform of the prediction.

        Parameters
        ----------
        wavenumbers : 1-D array
        intensity   : 1-D array
        nr_background : 1-D array, optional
            If supplied, the ratio I/I_NR is passed to the network.

        Returns
        -------
        dict with keys ``wavenumbers``, ``amplitude``, ``phase``,
        ``im_chi3``, ``re_chi3``.
        """
        wn = np.asarray(wavenumbers, dtype=float)
        I  = np.maximum(np.asarray(intensity, dtype=float), 0.0)

        if nr_background is not None:
            bg = np.maximum(nr_background, 1e-30)
            I  = I / bg

        # Resize to model's expected input length
        original_len = len(I)
        if len(I) != self.input_len:
            from scipy.interpolate import interp1d
            I = interp1d(
                np.linspace(0, 1, len(I)), I
            )(np.linspace(0, 1, self.input_len))

        x = I.copy()
        if self.normalize_input:
            mn, mx = x.min(), x.max()
            x = (x - mn) / (mx - mn + 1e-30)

        im_pred = self._infer(x)

        # Resize back if needed
        if len(im_pred) != original_len:
            from scipy.interpolate import interp1d
            im_pred = interp1d(
                np.linspace(0, 1, len(im_pred)), im_pred
            )(np.linspace(0, 1, original_len))

        # Estimate Re[χ³] via KK on the predicted Im[χ³]
        from scipy.signal import hilbert
        re_pred = -np.real(hilbert(im_pred))   # Hilbert of Im → -Re (KK)

        amplitude = np.sqrt(im_pred ** 2 + re_pred ** 2)
        phase     = np.arctan2(im_pred, re_pred)

        return {
            "wavenumbers": wn,
            "amplitude":   amplitude,
            "phase":       phase,
            "im_chi3":     im_pred,
            "re_chi3":     re_pred,
        }

    def _infer(self, x: np.ndarray) -> np.ndarray:
        if self.backend == "torch":
            return self._infer_torch(x)
        else:
            return self._infer_keras(x)

    def _infer_torch(self, x: np.ndarray) -> np.ndarray:
        import torch
        t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t = t.to(self._device)
        with torch.no_grad():
            out = self._model(t)
        return out.squeeze().cpu().numpy()

    def _infer_keras(self, x: np.ndarray) -> np.ndarray:
        t = x[np.newaxis, :, np.newaxis].astype(np.float32)
        out = self._model(t, training=False)
        return np.squeeze(out.numpy())

    # ── training / fine-tuning ────────────────────────────────────────────────
    def fine_tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-4,
        validation_split: float = 0.1,
        verbose: bool = True,
    ) -> "NeuralNetRetriever":
        """
        Fine-tune the loaded model on new labelled CARS data.

        Parameters
        ----------
        X_train : ndarray, shape (N, L)
            Input CARS spectra (N samples, L wavenumber points).
        y_train : ndarray, shape (N, L)
            Target Im[χ³] spectra.
        epochs, batch_size, lr : training hyper-parameters.
        validation_split : float
            Fraction held out for validation.

        Returns
        -------
        self  (for chaining)
        """
        if self.backend == "torch":
            self._fine_tune_torch(X_train, y_train, epochs, batch_size, lr,
                                  validation_split, verbose)
        else:
            self._fine_tune_keras(X_train, y_train, epochs, batch_size, lr,
                                  validation_split, verbose)
        return self

    def _fine_tune_torch(self, X, y, epochs, batch_size, lr, val_split, verbose):
        import torch
        from torch.utils.data import TensorDataset, DataLoader, random_split

        X_t = torch.tensor(X[:, np.newaxis, :], dtype=torch.float32)
        y_t = torch.tensor(y[:, np.newaxis, :], dtype=torch.float32)
        ds  = TensorDataset(X_t, y_t)

        n_val = max(1, int(len(ds) * val_split))
        train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size)

        opt  = torch.optim.Adam(self._model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        self._model.train()

        for ep in range(1, epochs + 1):
            train_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(self._device), yb.to(self._device)
                pred = self._model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_ds)

            val_loss = 0.0
            self._model.eval()
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(self._device), yb.to(self._device)
                    val_loss += loss_fn(self._model(xb), yb).item() * len(xb)
            val_loss /= len(val_ds)
            self._model.train()

            if verbose:
                print(f"Epoch {ep:3d}/{epochs}  "
                      f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        self._model.eval()

    def _fine_tune_keras(self, X, y, epochs, batch_size, lr, val_split, verbose):
        import tensorflow as tf
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="mse",
        )
        X_in = X[:, :, np.newaxis].astype(np.float32)
        y_in = y[:, :, np.newaxis].astype(np.float32)
        self._model.fit(
            X_in, y_in,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            verbose=1 if verbose else 0,
        )

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Save the current model weights."""
        path = Path(path)
        if self.backend == "torch":
            import torch
            torch.save(self._model.state_dict(), path)
        else:
            self._model.save(str(path))
        print(f"Model saved to {path}")

    def __repr__(self) -> str:
        return (
            f"NeuralNetRetriever(model_name={self.model_name!r}, "
            f"backend={self.backend!r}, architecture={self.architecture!r})"
        )
