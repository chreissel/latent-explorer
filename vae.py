"""
The one engine.

A single, configurable convolutional Variational Autoencoder (VAE). The SAME
class is used for both the MNIST "Digits" model and the Galaxy10 "Galaxies"
model -- only the number of input channels, the image size, and the latent
dimensionality differ. That is the whole teaching point of the booth: it is the
identical machine, learning two different things.

A VAE squeezes an image down to a handful of numbers (the "latent vector"),
then rebuilds the image from just those numbers. The space of all possible
latent vectors is the "latent space" -- a smooth map of everything the model
has learned to imagine.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """A small conv VAE that adapts to any square image size and channel count.

    The encoder is a stack of stride-2 convolutions; the decoder mirrors it and
    uses exact-size interpolation on the way back up, so it reconstructs any
    input resolution (28x28 for digits, 69x69 for galaxies) without fiddly
    transposed-convolution arithmetic.
    """

    def __init__(
        self,
        img_channels: int,
        img_size: int,
        latent_dim: int,
        hidden_channels: tuple[int, ...] = (32, 64, 128),
    ) -> None:
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.hidden_channels = tuple(hidden_channels)

        # --- Encoder: stride-2 conv blocks -------------------------------
        enc_layers: list[nn.Module] = []
        in_c = img_channels
        for out_c in hidden_channels:
            enc_layers += [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_c = out_c
        self.encoder = nn.Sequential(*enc_layers)

        # Discover the spatial sizes at every stage with one dummy pass. We
        # store them so the decoder can interpolate back to the exact sizes.
        self._spatial_sizes = self._trace_spatial_sizes(img_size)
        bottleneck = self._spatial_sizes[-1]
        flat = hidden_channels[-1] * bottleneck * bottleneck
        self._flat_dim = flat
        self._bottleneck = bottleneck

        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat)

        # --- Decoder: mirror of the encoder, upsampling to exact sizes ----
        dec_layers: list[nn.Module] = []
        rev = list(reversed(hidden_channels))
        for i in range(len(rev)):
            in_c = rev[i]
            out_c = rev[i + 1] if i + 1 < len(rev) else hidden_channels[0]
            dec_layers += [
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.decoder = nn.Sequential(*dec_layers)
        self.final_conv = nn.Conv2d(
            hidden_channels[0], img_channels, kernel_size=3, stride=1, padding=1
        )

        # Target sizes for each decoder upsampling step (coarse -> fine),
        # ending at the original image size.
        self._decode_sizes = list(reversed(self._spatial_sizes[:-1])) + [img_size]

    def _trace_spatial_sizes(self, img_size: int) -> list[int]:
        sizes = [img_size]
        s = img_size
        for _ in self.hidden_channels:
            s = (s + 2 * 1 - 4) // 2 + 1  # conv: kernel 4, stride 2, pad 1
            sizes.append(s)
        return sizes

    # -- core VAE pieces --------------------------------------------------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        c = self.hidden_channels[-1]
        h = h.view(-1, c, self._bottleneck, self._bottleneck)
        # Walk up the decoder, resizing to the exact size before each block.
        n_blocks = len(self.hidden_channels)
        for i in range(n_blocks):
            target = self._decode_sizes[i]
            h = F.interpolate(h, size=(target, target), mode="nearest")
            block = self.decoder[i * 3 : i * 3 + 3]
            h = block(h)
        return torch.sigmoid(self.final_conv(h))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruction (BCE) + beta * KL divergence, averaged per image."""
    bce = F.binary_cross_entropy(recon, target, reduction="sum") / target.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    return bce + beta * kld, bce, kld


# Model configurations: SAME engine, two settings. ----------------------
MODEL_CONFIGS = {
    "digits": {
        "img_channels": 1,
        "img_size": 28,
        "latent_dim": 2,
        "hidden_channels": (32, 64, 128),
        "weights": "models/digits.pt",
        "display_name": "Digits (MNIST)",
    },
    "galaxies": {
        "img_channels": 3,
        "img_size": 69,
        "latent_dim": 24,
        "hidden_channels": (32, 64, 128, 256),
        "weights": "models/galaxies.pt",
        "display_name": "Galaxies (Galaxy10)",
    },
    "gravityspy": {
        "img_channels": 1,          # Q-transform spectrograms (intensity)
        "img_size": 64,
        "latent_dim": 32,
        "hidden_channels": (32, 64, 128, 256),
        "weights": "models/gravityspy.pt",
        "display_name": "Gravity Spy (LIGO glitches)",
        "colormap": "viridis",      # colourise the grayscale spectrograms
    },
}


def build_vae(config: dict) -> VAE:
    return VAE(
        img_channels=config["img_channels"],
        img_size=config["img_size"],
        latent_dim=config["latent_dim"],
        hidden_channels=config["hidden_channels"],
    )
