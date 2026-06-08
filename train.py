"""
Train the latent-explorer VAEs.

Same code path for both models -- only the dataset name and the config (channels,
image size, latent dim) change. Weights are saved to ./models so the booth app
loads instantly without retraining.

Usage:
    python train.py              # train both: digits then galaxies
    python train.py --dataset digits
    python train.py --dataset galaxies --epochs 30 --latent-dim 24

Rough CPU training times (4 cores): digits ~1-2 min, galaxies ~3-6 min.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

import data as datamod
from vae import MODEL_CONFIGS, build_vae, vae_loss

MODELS_DIR = "models"
N_SAMPLE_BANK = 256  # real images stashed in the checkpoint for interpolation mode


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one(
    name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    latent_dim: int | None,
    beta: float,
    max_images: int | None,
    synthetic: bool = False,
) -> None:
    device = get_device()
    config = dict(MODEL_CONFIGS[name])
    if latent_dim is not None:
        config["latent_dim"] = latent_dim

    print(f"\n=== Training '{name}' ({config['display_name']}) on {device} ===")
    print(f"    img {config['img_channels']}x{config['img_size']}x{config['img_size']}, "
          f"latent_dim={config['latent_dim']}, beta={beta}")

    x = datamod.load_dataset(name, max_images=max_images, synthetic=synthetic)
    src = "SYNTHETIC stand-in" if (synthetic and name == "galaxies") else "real"
    print(f"    dataset: {tuple(x.shape)}  [{src}]")

    loader = DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    model = build_vae(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tot, tot_bce, tot_kld, n = 0.0, 0.0, 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(batch)
            loss, bce, kld = vae_loss(recon, batch, mu, logvar, beta=beta)
            loss.backward()
            opt.step()
            tot += loss.item()
            tot_bce += bce.item()
            tot_kld += kld.item()
            n += 1
        dt = time.time() - t0
        print(f"  epoch {epoch:3d}/{epochs}  loss={tot/n:8.2f}  "
              f"bce={tot_bce/n:8.2f}  kld={tot_kld/n:6.2f}  ({dt:4.1f}s)")

    # Stash a bank of real images for the interpolation mode (offline-friendly).
    idx = torch.randperm(x.size(0))[:N_SAMPLE_BANK]
    sample_bank = x[idx].clone().cpu()

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = config["weights"]
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "sample_bank": sample_bank,
        },
        out_path,
    )
    print(f"    saved -> {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description="Train latent-explorer VAEs")
    p.add_argument("--dataset", choices=["digits", "galaxies", "both"],
                   default="both")
    p.add_argument("--epochs", type=int, default=None,
                   help="override epochs (default: 20 digits / 40 galaxies)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=None,
                   help="override latent dimensionality")
    p.add_argument("--beta", type=float, default=None,
                   help="KL weight (default: 1.0 digits / 2.0 galaxies)")
    p.add_argument("--max-images", type=int, default=None,
                   help="cap galaxy images (for quick tests)")
    p.add_argument("--synthetic-galaxies", action="store_true",
                   help="train galaxies on an offline procedurally-generated "
                        "stand-in instead of downloading real Galaxy10")
    args = p.parse_args()

    targets = ["digits", "galaxies"] if args.dataset == "both" else [args.dataset]
    defaults = {
        "digits": {"epochs": 20, "beta": 1.0},
        "galaxies": {"epochs": 40, "beta": 2.0},
    }
    for name in targets:
        train_one(
            name=name,
            epochs=args.epochs or defaults[name]["epochs"],
            batch_size=args.batch_size,
            lr=args.lr,
            latent_dim=args.latent_dim,
            beta=args.beta if args.beta is not None else defaults[name]["beta"],
            max_images=args.max_images,
            synthetic=args.synthetic_galaxies,
        )


if __name__ == "__main__":
    main()
