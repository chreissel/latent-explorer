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

    Uses the local cache only (download=False) so it works fully offline at the
    booth. Raises if MNIST hasn't been downloaded yet (train.py fetches it).
    """
    from torchvision import datasets

    ds = datasets.MNIST(root=DATA_DIR, train=train, download=False)
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


# --- Synthetic galaxy stand-in ------------------------------------------
# A procedurally-generated "galaxy-like" dataset (fuzzy elliptical bulges with
# optional spiral arms, varied colour/orientation/brightness) at the SAME shape
# as Galaxy10 (3 x 69 x 69). It lets you smoke-test the galaxy pipeline and the
# app's Morph tab fully OFFLINE -- no 200 MB download -- when the real dataset
# is unavailable. It is NOT a substitute for real Galaxy10 at the booth; train
# on the real data there for recognizable reconstructions.
def synthetic_galaxies(n: int = 6000, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    S = 69
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    cx = cy = (S - 1) / 2.0
    out = np.zeros((n, S, S, 3), dtype=np.float32)
    for i in range(n):
        ax = rng.uniform(6, 16)              # bulge radii
        ay = ax * rng.uniform(0.4, 1.0)
        ang = rng.uniform(0, np.pi)
        ca, sa = np.cos(ang), np.sin(ang)
        x = (xx - cx) * ca + (yy - cy) * sa
        y = -(xx - cx) * sa + (yy - cy) * ca
        r2 = (x / ax) ** 2 + (y / ay) ** 2
        bulge = np.exp(-r2)                   # smooth elliptical bulge
        img = bulge.copy()
        if rng.random() < 0.5:               # add spiral arms for ~half
            r = np.sqrt(x ** 2 + y ** 2) + 1e-3
            theta = np.arctan2(y, x)
            tightness = rng.uniform(0.3, 0.7)
            arms = rng.integers(2, 4)
            spiral = 0.5 * (1 + np.cos(arms * theta - r * tightness))
            img += 0.6 * spiral * np.exp(-r / rng.uniform(10, 20))
        img = img / (img.max() + 1e-6)
        # warm-to-cool colour tint per galaxy
        tint = rng.uniform(0.6, 1.0, size=3)
        tint = tint / tint.max()
        colored = img[..., None] * tint[None, None, :]
        colored += rng.normal(0, 0.02, size=colored.shape)  # faint noise
        out[i] = np.clip(colored, 0, 1)
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()


def load_dataset(
    name: str, max_images: int | None = None, synthetic: bool = False
) -> torch.Tensor:
    if name == "digits":
        return load_mnist(train=True)
    if name == "galaxies":
        if synthetic:
            return synthetic_galaxies(n=max_images or 6000)
        return load_galaxy10(max_images=max_images)
    raise ValueError(f"Unknown dataset: {name}")
