"""
Dataset loading and caching.

Two datasets, both cached under ./data so the booth runs fully offline after a
one-time download:

  * Digits   -> MNIST via torchvision (auto-downloads).
  * Galaxies -> Galaxy10 (SDSS, 69x69x3, 10 classes). We fetch the .h5 directly
                rather than pulling in the heavy astroNN/TensorFlow stack, which
                keeps the install small and laptop-friendly.
"""

from __future__ import annotations

import os

import numpy as np
import torch

# Where datasets are downloaded/cached. Override without code changes by setting
# the LATENT_DATA_DIR environment variable (e.g. a big external disk).
DATA_DIR = os.environ.get("LATENT_DATA_DIR", "data")
GALAXY10_PATH = os.path.join(DATA_DIR, "Galaxy10.h5")

# Mirrors for the Galaxy10 SDSS .h5 (the original 69x69x3 version used by
# astroNN's load_galaxy10()). Tried in order. Zenodo is the canonical home;
# the utoronto links are the historical astroNN mirrors.
GALAXY10_URLS = [
    "https://zenodo.org/records/10844811/files/Galaxy10.h5?download=1",
    "https://astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10.h5",
    "https://www.astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10.h5",
]

# Gravity Spy training set (LIGO glitch Q-transform spectrograms), 8535 samples,
# 22 classes, 4 durations each. Single .h5 hosted on Zenodo.
GRAVITYSPY_PATH = os.path.join(DATA_DIR, "trainingsetv1d1.h5")
GRAVITYSPY_URLS = [
    "https://zenodo.org/records/1486046/files/trainingsetv1d1.h5?download=1",
    "https://zenodo.org/records/1476551/files/trainingsetv1d1.h5?download=1",
]
GRAVITYSPY_SIZE = 64          # spectrograms resized to this square for the VAE


# --- MNIST ---------------------------------------------------------------
def load_mnist(train: bool = True) -> torch.Tensor:
    """Return MNIST images as a float tensor of shape (N, 1, 28, 28) in [0, 1]."""
    from torchvision import datasets, transforms

    ds = datasets.MNIST(
        root=DATA_DIR,
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )
    x = ds.data.float().unsqueeze(1) / 255.0
    return x


def load_mnist_labeled(n: int | None = None, train: bool = True):
    """Return (images, labels): images (N,1,28,28) in [0,1], labels (N,) 0-9.

    Loads from the DATA_DIR cache (set LATENT_DATA_DIR to point at an existing
    copy); downloads MNIST there if it isn't already present so the class-region
    map still builds even when the cache lives elsewhere.
    """
    from torchvision import datasets

    ds = datasets.MNIST(root=DATA_DIR, train=train, download=True)
    x = ds.data.float().unsqueeze(1) / 255.0
    y = ds.targets.clone()
    if n is not None and n < x.size(0):
        idx = torch.randperm(x.size(0))[:n]
        x, y = x[idx], y[idx]
    return x, y


# --- Galaxy10 ------------------------------------------------------------
def download_galaxy10() -> str:
    """Download the Galaxy10 .h5 to ./data if not already cached."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(GALAXY10_PATH) and os.path.getsize(GALAXY10_PATH) > 1_000_000:
        return GALAXY10_PATH

    import requests

    last_err: Exception | None = None
    for url in GALAXY10_URLS:
        try:
            print(f"Downloading Galaxy10 from {url} ...")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                tmp = GALAXY10_PATH + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = 100 * done / total
                            print(f"\r  {done/1e6:6.1f} MB / {total/1e6:6.1f} MB "
                                  f"({pct:5.1f}%)", end="", flush=True)
                print()
                os.replace(tmp, GALAXY10_PATH)
            return GALAXY10_PATH
        except Exception as e:  # try the next mirror
            last_err = e
            print(f"  failed: {e}")

    raise RuntimeError(
        "Could not download Galaxy10 from any mirror. Download Galaxy10.h5 "
        f"manually into ./{DATA_DIR}/. Last error: {last_err}"
    )


def load_galaxy10(max_images: int | None = None) -> torch.Tensor:
    """Return Galaxy10 images as a float tensor (N, 3, 69, 69) in [0, 1]."""
    import h5py

    path = download_galaxy10()
    with h5py.File(path, "r") as f:
        images = np.asarray(f["images"])  # (N, 69, 69, 3), uint8
        if max_images is not None:
            images = images[:max_images]
    x = torch.from_numpy(images).float() / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()  # -> (N, 3, 69, 69)
    return x


def load_galaxy10_labeled(n: int | None = None):
    """Return (images, labels): images (N,3,69,69) in [0,1], labels (N,) 0-9
    (Galaxy10 morphology classes). Cached .h5 is downloaded if needed."""
    import h5py

    path = download_galaxy10()
    with h5py.File(path, "r") as f:
        images = np.asarray(f["images"])          # (N, 69, 69, 3) uint8
        labels = np.asarray(f["ans"]).astype(np.int64)  # (N,) 0-9
    x = torch.from_numpy(images).float() / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()
    y = torch.from_numpy(labels)
    if n is not None and n < x.size(0):
        idx = torch.randperm(x.size(0))[:n]
        x, y = x[idx], y[idx]
    return x, y


# --- Gravity Spy (LIGO glitch spectrograms) -----------------------------
def _download(name: str, urls: list, dest: str) -> str:
    """Download `dest` from the first working mirror in `urls` (with progress)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return dest
    import requests

    last_err: Exception | None = None
    for url in urls:
        try:
            print(f"Downloading {name} from {url} ...")
            with requests.get(url, stream=True, timeout=90) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {done/1e6:6.1f} MB / {total/1e6:6.1f} MB "
                                  f"({100*done/total:5.1f}%)", end="", flush=True)
                print()
                os.replace(tmp, dest)
            return dest
        except Exception as e:
            last_err = e
            print(f"  failed: {e}")
    raise RuntimeError(
        f"Could not download {name} from any mirror. Place the file manually at "
        f"{dest}. Last error: {last_err}"
    )


def download_gravityspy() -> str:
    return _download("Gravity Spy", GRAVITYSPY_URLS, GRAVITYSPY_PATH)


def _gravityspy_walk(f, duration="2.0"):
    """Yield (image_array, label_name) for one duration from the Gravity Spy h5.

    Layout: /<label>/<train|validation|test>/<gravityspy_id>/<duration>.png,
    each dataset shaped (1, 140, 170).
    """
    for label in f.keys():
        g = f[label]
        if not hasattr(g, "keys"):
            continue
        for stype in g.keys():
            gs = g[stype]
            if not hasattr(gs, "keys"):
                continue
            for gid in gs.keys():
                node = gs[gid]
                if not hasattr(node, "keys"):
                    continue
                key = f"{duration}.png"
                if key not in node:
                    imgs = [k for k in node.keys()]
                    if not imgs:
                        continue
                    key = imgs[len(imgs) // 2]
                yield np.asarray(node[key], dtype=np.float32), label


def _prep_spectrogram(arr: np.ndarray) -> torch.Tensor:
    """(1,H,W)/(H,W)/(H,W,3) intensity array -> (1, SIZE, SIZE) in [0,1]."""
    import torch.nn.functional as F

    if arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].mean(-1)[None]      # RGB spectrogram -> intensity
    elif arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = arr[:3].mean(0)[None]
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if float(t.max()) > 1.0:
        t = t / 255.0
    t = F.interpolate(t.unsqueeze(0), size=(GRAVITYSPY_SIZE, GRAVITYSPY_SIZE),
                      mode="bilinear", align_corners=False).squeeze(0)
    return t.clamp(0, 1)


def load_gravityspy_labeled(n: int | None = None):
    """Return (images, labels): images (N,1,64,64) in [0,1], labels (N,) glitch
    class indices (0..21). Downloads the Gravity Spy .h5 if not cached."""
    import h5py

    path = download_gravityspy()
    imgs, labs, label_names = [], [], {}
    with h5py.File(path, "r") as f:
        names = sorted(k for k in f.keys() if hasattr(f[k], "keys"))
        label_names = {name: i for i, name in enumerate(names)}
        for arr, label in _gravityspy_walk(f):
            imgs.append(_prep_spectrogram(arr))
            labs.append(label_names[label])
    x = torch.stack(imgs)
    y = torch.tensor(labs, dtype=torch.long)
    if n is not None and n < x.size(0):
        idx = torch.randperm(x.size(0))[:n]
        x, y = x[idx], y[idx]
    return x, y


def load_gravityspy(max_images: int | None = None) -> torch.Tensor:
    x, _ = load_gravityspy_labeled(n=max_images)
    return x


# --- Synthetic galaxy stand-in ------------------------------------------
# A procedurally-generated "galaxy-like" dataset (fuzzy elliptical bulges with
# optional spiral arms, varied colour/orientation/brightness) at the SAME shape
# as Galaxy10 (3 x 69 x 69). It lets you smoke-test the galaxy pipeline and the
# app's Morph tab fully OFFLINE -- no 200 MB download -- when the real dataset
# is unavailable. It is NOT a substitute for real Galaxy10 at the booth; train
# on the real data there for recognizable reconstructions.
def synthetic_galaxies(n: int = 6000, seed: int = 0, return_labels: bool = False):
    rng = np.random.default_rng(seed)
    S = 69
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    cx = cy = (S - 1) / 2.0
    out = np.zeros((n, S, S, 3), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ax = rng.uniform(6, 16)              # bulge radii
        elong = rng.uniform(0.4, 1.0)
        ay = ax * elong
        ang = rng.uniform(0, np.pi)
        ca, sa = np.cos(ang), np.sin(ang)
        x = (xx - cx) * ca + (yy - cy) * sa
        y = -(xx - cx) * sa + (yy - cy) * ca
        r2 = (x / ax) ** 2 + (y / ay) ** 2
        bulge = np.exp(-r2)                   # smooth elliptical bulge
        img = bulge.copy()
        spiral_on = rng.random() < 0.5
        arms = int(rng.integers(2, 4))
        if spiral_on:                         # add spiral arms for ~half
            r = np.sqrt(x ** 2 + y ** 2) + 1e-3
            theta = np.arctan2(y, x)
            tightness = rng.uniform(0.3, 0.7)
            spiral = 0.5 * (1 + np.cos(arms * theta - r * tightness))
            img += 0.6 * spiral * np.exp(-r / rng.uniform(10, 20))
        img = img / (img.max() + 1e-6)
        # warm-to-cool colour tint per galaxy
        tint = rng.uniform(0.6, 1.0, size=3)
        tint = tint / tint.max()
        colored = img[..., None] * tint[None, None, :]
        colored += rng.normal(0, 0.02, size=colored.shape)  # faint noise
        out[i] = np.clip(colored, 0, 1)
        # Pseudo "morphology class" so the offline stand-in has structure to map:
        # 0-2 = elliptical (round..elongated), 3-4 = spiral (2..3 arms).
        if spiral_on:
            labels[i] = 3 + (arms - 2)
        else:
            labels[i] = 0 if elong > 0.8 else (1 if elong > 0.6 else 2)
    x = torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()
    if return_labels:
        return x, torch.from_numpy(labels)
    return x


# --- Synthetic Gravity Spy stand-in -------------------------------------
# Procedurally-generated glitch-like spectrograms (grayscale, 64x64) with a few
# distinct morphologies, so the Gravity Spy pipeline and app can be tested fully
# OFFLINE without the Zenodo download. NOT real LIGO data.
def synthetic_spectrograms(n: int = 6000, seed: int = 0,
                           return_labels: bool = False):
    rng = np.random.default_rng(seed)
    S = GRAVITYSPY_SIZE
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    out = np.zeros((n, 1, S, S), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        c = int(rng.integers(0, 6))
        img = np.zeros((S, S), dtype=np.float32)
        if c == 0:                                   # Blip: short vertical streak
            cx = rng.uniform(0.3, 0.7) * S
            img += np.exp(-((xx - cx) ** 2) / (2 * rng.uniform(2, 4) ** 2))
        elif c == 1:                                 # Whistle: rising chirp curve
            f = 0.15 + 0.7 * (xx / S) ** 2
            img += np.exp(-((yy / S - f) ** 2) / (2 * 0.004))
        elif c == 2:                                 # Scattered light: stacked arches
            for k in range(1, 4):
                arch = 0.5 * S + 0.12 * S * k * np.sin(np.pi * xx / S)
                img += np.exp(-((yy - arch) ** 2) / (2 * 3.0 ** 2))
        elif c == 3:                                 # Koi fish: central blob
            img += np.exp(-(((xx - S / 2) ** 2 + (yy - S / 2) ** 2)) /
                          (2 * rng.uniform(6, 10) ** 2))
        elif c == 4:                                 # Line: horizontal band
            ly = rng.uniform(0.3, 0.7) * S
            img += np.exp(-((yy - ly) ** 2) / (2 * rng.uniform(2, 4) ** 2))
        else:                                        # Low-freq: bottom blob
            img += np.exp(-((xx - S / 2) ** 2 + (yy - 0.8 * S) ** 2) /
                          (2 * rng.uniform(8, 12) ** 2))
        img += rng.normal(0, 0.03, size=img.shape)
        out[i, 0] = np.clip(img / (img.max() + 1e-6), 0, 1)
        labels[i] = c
    x = torch.from_numpy(out)
    if return_labels:
        return x, torch.from_numpy(labels)
    return x


def load_dataset(
    name: str, max_images: int | None = None, synthetic: bool = False
) -> torch.Tensor:
    if name == "digits":
        return load_mnist(train=True)
    if name == "galaxies":
        if synthetic:
            return synthetic_galaxies(n=max_images or 6000)
        return load_galaxy10(max_images=max_images)
    if name == "gravityspy":
        if synthetic:
            return synthetic_spectrograms(n=max_images or 6000)
        return load_gravityspy(max_images=max_images)
    raise ValueError(f"Unknown dataset: {name}")


def load_labeled(name: str, n: int | None = None):
    """(images, labels) for building a latent class-map. Real datasets fall back
    to their labelled synthetic stand-in if the download can't be fetched."""
    if name == "digits":
        return load_mnist_labeled(n=n)
    if name == "galaxies":
        try:
            return load_galaxy10_labeled(n=n)
        except Exception:
            return synthetic_galaxies(n=n or 6000, return_labels=True)
    if name == "gravityspy":
        try:
            return load_gravityspy_labeled(n=n)
        except Exception:
            return synthetic_spectrograms(n=n or 6000, return_labels=True)
    raise ValueError(f"Unknown dataset: {name}")
