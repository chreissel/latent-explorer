"""
latent-explorer booth app.

Loads the two trained VAEs and launches a touch-friendly Gradio interface for a
science-fair booth. Runs fully offline once the weights exist in ./models.

Two ways to play:
  * Digit Explorer  -- click anywhere on the 2D latent "map" and watch the digit
                       the model imagines at that spot. The whole map is shown as
                       one big grid: every digit the model can dream up.
  * Morph / Blend   -- pick two real images (digits or galaxies) and slide to
                       blend smoothly from one into the other through latent space.
"""

from __future__ import annotations

import os

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

from vae import MODEL_CONFIGS, build_vae

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Display sizing for the booth screen.
BIG = 384            # size of the main decoded image (px)
THUMB = 140          # endpoint thumbnail size (px)
GRID_N = 20          # digit latent map is GRID_N x GRID_N cells
LATENT_RANGE = 3.0   # explore latent coords in [-RANGE, +RANGE]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
class LoadedModel:
    def __init__(self, name: str):
        config = dict(MODEL_CONFIGS[name])
        path = config["weights"]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing weights for '{name}': {path}\n"
                f"Train first:  python train.py --dataset {name}"
            )
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        # Honor the config that was actually trained (e.g. latent dim override).
        config.update(ckpt.get("config", {}))
        self.config = config
        self.name = name
        self.model = build_vae(config).to(DEVICE)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.sample_bank: torch.Tensor = ckpt["sample_bank"]
        self.channels = config["img_channels"]
        self.size = config["img_size"]
        self.latent_dim = config["latent_dim"]


MODELS: dict[str, LoadedModel] = {}


def get_model(name: str) -> LoadedModel:
    if name not in MODELS:
        MODELS[name] = LoadedModel(name)
    return MODELS[name]


# ---------------------------------------------------------------------------
# Tensor <-> image helpers
# ---------------------------------------------------------------------------
def tensor_to_pil(img: torch.Tensor, size: int, smooth: bool) -> Image.Image:
    """(C,H,W) float in [0,1] -> upscaled PIL image."""
    arr = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    if arr.shape[0] == 1:
        pil = Image.fromarray(arr[0], mode="L")
    else:
        pil = Image.fromarray(np.transpose(arr, (1, 2, 0)), mode="RGB")
    resample = Image.LANCZOS if smooth else Image.NEAREST
    return pil.resize((size, size), resample)


@torch.no_grad()
def decode(lm: LoadedModel, z: torch.Tensor) -> torch.Tensor:
    return lm.model.decode(z.to(DEVICE)).squeeze(0).cpu()


@torch.no_grad()
def encode_mu(lm: LoadedModel, img: torch.Tensor) -> torch.Tensor:
    mu, _ = lm.model.encode(img.unsqueeze(0).to(DEVICE))
    return mu.squeeze(0).cpu()


# ---------------------------------------------------------------------------
# Digit Explorer: the 2D latent map + clickable pad
# ---------------------------------------------------------------------------
def build_latent_grid(lm: LoadedModel) -> Image.Image:
    """Decode a GRID_N x GRID_N sweep of the 2D latent space into one image."""
    coords = np.linspace(-LATENT_RANGE, LATENT_RANGE, GRID_N)
    zs = []
    for yi in coords[::-1]:        # top row = high y
        for xi in coords:
            zs.append([xi, yi])
    z = torch.tensor(zs, dtype=torch.float32)
    with torch.no_grad():
        imgs = lm.model.decode(z.to(DEVICE)).cpu()  # (N,1,28,28)
    cell = lm.size
    grid = Image.new("L", (GRID_N * cell, GRID_N * cell))
    for k in range(GRID_N * GRID_N):
        tile = (imgs[k, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)
        grid.paste(Image.fromarray(tile, mode="L"),
                   ((k % GRID_N) * cell, (k // GRID_N) * cell))
    return grid


_GRID_CACHE: dict[str, Image.Image] = {}


def get_grid(lm: LoadedModel) -> Image.Image:
    if lm.name not in _GRID_CACHE:
        _GRID_CACHE[lm.name] = build_latent_grid(lm)
    return _GRID_CACHE[lm.name]


# --- Class-region map ---------------------------------------------------
# Instead of tiling decoded images (which looks like a plain image grid), we
# colour the latent plane by *which digit class lives where*. We encode a batch
# of labelled MNIST digits to their 2-D latent positions, fit a simple
# per-class Gaussian (QDA), and paint each point of the plane with the colour of
# its most likely class. Boundaries between colours are exactly the "in-between"
# regions the user can click to explore blends of two digits.
DIGIT_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207),
]


def _load_font(size: int):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


def build_class_region_map(lm: LoadedModel, size: int = 560,
                           n_fit: int = 8000) -> Image.Image:
    """Colour the 2-D latent plane by most-likely digit class (QDA)."""
    import data as datamod

    imgs, labels = datamod.load_mnist_labeled(n=n_fit)
    with torch.no_grad():
        mu, _ = lm.model.encode(imgs.to(DEVICE))
    pts = mu.cpu().numpy()
    labels = labels.numpy()

    # Per-class Gaussian parameters.
    means, invs, logdets, logpri, classes = {}, {}, {}, {}, []
    for c in range(10):
        Xc = pts[labels == c]
        if len(Xc) < 5:
            continue
        m = Xc.mean(0)
        cov = np.cov(Xc.T) + np.eye(2) * 1e-3
        means[c] = m
        invs[c] = np.linalg.inv(cov)
        logdets[c] = float(np.log(np.linalg.det(cov)))
        logpri[c] = float(np.log(len(Xc) / len(pts)))
        classes.append(c)

    # Grid of latent coords over the plane (top row = high y).
    W = H = size
    xs = np.linspace(-LATENT_RANGE, LATENT_RANGE, W)
    ys = np.linspace(LATENT_RANGE, -LATENT_RANGE, H)
    gx, gy = np.meshgrid(xs, ys)
    G = np.stack([gx.ravel(), gy.ravel()], axis=1)

    scores = np.empty((len(classes), G.shape[0]), dtype=np.float64)
    for i, c in enumerate(classes):
        d = G - means[c]
        q = np.einsum("pi,ij,pj->p", d, invs[c], d)
        scores[i] = -0.5 * q - 0.5 * logdets[c] + logpri[c]
    cls = np.array(classes)[np.argmax(scores, axis=0)].reshape(H, W)

    # Paint lightened class colours so numerals/boundaries stay legible.
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for c in classes:
        base = np.array(DIGIT_COLORS[c], dtype=np.float32)
        arr[cls == c] = (base * 0.6 + 255 * 0.4).astype(np.uint8)

    # Dark boundaries where the predicted class changes.
    bound = np.zeros((H, W), dtype=bool)
    bound[:-1, :] |= cls[:-1, :] != cls[1:, :]
    bound[:, :-1] |= cls[:, :-1] != cls[:, 1:]
    arr[bound] = (30, 30, 30)

    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(38)
    for c in classes:
        mx, my = means[c]
        px = int((mx + LATENT_RANGE) / (2 * LATENT_RANGE) * W)
        py = int((LATENT_RANGE - my) / (2 * LATENT_RANGE) * H)
        px = min(max(px, 18), W - 18)
        py = min(max(py, 18), H - 18)
        for ox, oy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            draw.text((px + ox, py + oy), str(c), fill=(0, 0, 0),
                      font=font, anchor="mm")
        draw.text((px, py), str(c), fill=(255, 255, 255), font=font, anchor="mm")
    return img


_PAD_CACHE: dict[str, Image.Image] = {}


def get_pad_base(lm: LoadedModel) -> Image.Image:
    """The clickable pad background: class-region map, or decoded grid if the
    labelled MNIST cache isn't available."""
    if lm.name not in _PAD_CACHE:
        try:
            _PAD_CACHE[lm.name] = build_class_region_map(lm)
        except Exception as e:
            print(f"class-region map unavailable ({e}); using decoded grid")
            _PAD_CACHE[lm.name] = build_latent_grid(lm).convert("RGB")
    return _PAD_CACHE[lm.name]


def pad_with_marker(lm: LoadedModel, lx: float, ly: float) -> Image.Image:
    """Latent map (RGB) with a crosshair drawn at latent coord (lx, ly)."""
    grid = get_pad_base(lm).copy()
    W, H = grid.size
    px = int((lx + LATENT_RANGE) / (2 * LATENT_RANGE) * W)
    py = int((LATENT_RANGE - ly) / (2 * LATENT_RANGE) * H)
    d = ImageDraw.Draw(grid)
    r = 11
    d.ellipse([px - r, py - r, px + r, py + r], outline=(255, 255, 255), width=5)
    d.line([(px - r, py), (px + r, py)], fill=(255, 60, 60), width=3)
    d.line([(px, py - r), (px, py + r)], fill=(255, 60, 60), width=3)
    d.ellipse([px - r, py - r, px + r, py + r], outline=(255, 60, 60), width=3)
    return grid


def digit_from_latent(lx: float, ly: float):
    lm = get_model("digits")
    z = torch.tensor([[lx, ly]], dtype=torch.float32)
    img = decode(lm, z)
    return (tensor_to_pil(img, BIG, smooth=False),
            pad_with_marker(lm, lx, ly),
            f"latent = ({lx:+.2f}, {ly:+.2f})")


def on_pad_click(evt: gr.SelectData):
    lm = get_model("digits")
    grid = get_pad_base(lm)
    W, H = grid.size
    px, py = evt.index  # pixel coords on the displayed pad
    lx = px / W * (2 * LATENT_RANGE) - LATENT_RANGE
    ly = LATENT_RANGE - py / H * (2 * LATENT_RANGE)
    out_img, pad_img, label = digit_from_latent(lx, ly)
    return out_img, pad_img, label, round(lx, 2), round(ly, 2)


# ---------------------------------------------------------------------------
# Morph / Interpolation (both datasets)
# ---------------------------------------------------------------------------
def random_endpoints(dataset: str):
    lm = get_model(dataset)
    n = lm.sample_bank.size(0)
    ia, ib = np.random.choice(n, size=2, replace=False)
    za = encode_mu(lm, lm.sample_bank[ia])
    zb = encode_mu(lm, lm.sample_bank[ib])
    smooth = lm.channels == 3
    thumb_a = tensor_to_pil(lm.sample_bank[ia], THUMB, smooth)
    thumb_b = tensor_to_pil(lm.sample_bank[ib], THUMB, smooth)
    blended = interp_image(lm, za, zb, 0.5)
    return za, zb, thumb_a, thumb_b, blended, 0.5


def interp_image(lm: LoadedModel, za: torch.Tensor, zb: torch.Tensor, t: float):
    z = (1 - t) * za + t * zb
    img = decode(lm, z.unsqueeze(0))
    return tensor_to_pil(img, BIG, smooth=lm.channels == 3)


def on_blend(dataset: str, za, zb, t: float):
    lm = get_model(dataset)
    if za is None or zb is None:
        return None
    return interp_image(lm, za, zb, t)


def on_dataset_change(dataset: str):
    # Fresh endpoints whenever the dataset switches.
    za, zb, ta, tb, blended, t = random_endpoints(dataset)
    return za, zb, ta, tb, blended, t


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Image generation", theme=gr.themes.Soft(
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"])) as demo:
        gr.Markdown("# Image generation")

        with gr.Tabs():
            # --- Digit Explorer ---------------------------------------
            with gr.Tab("✏️  Image generation"):
                with gr.Row():
                    pad = gr.Image(label="Input — tap to explore",
                                   interactive=False, height=480,
                                   show_download_button=False)
                    out_digit = gr.Image(label="Generated Image", height=480,
                                         show_download_button=False)
                coord_lbl = gr.Markdown()
                # X and Y sliders stacked vertically (below each other).
                sx = gr.Slider(-LATENT_RANGE, LATENT_RANGE, value=0.0,
                               step=0.05, label="latent X")
                sy = gr.Slider(-LATENT_RANGE, LATENT_RANGE, value=0.0,
                               step=0.05, label="latent Y")

                pad.select(on_pad_click, None,
                           [out_digit, pad, coord_lbl, sx, sy])
                for s in (sx, sy):
                    s.release(digit_from_latent, [sx, sy],
                              [out_digit, pad, coord_lbl])

            # --- Morph / Blend ----------------------------------------
            with gr.Tab("🔀  Morphing"):
                dataset = gr.Radio(
                    choices=[("Digits", "digits"), ("Galaxies", "galaxies")],
                    value="digits", label="Dataset")
                with gr.Row():
                    thumb_a = gr.Image(label="Start", height=THUMB + 40,
                                       interactive=False,
                                       show_download_button=False)
                    blended = gr.Image(label="Blend", height=BIG,
                                       show_download_button=False)
                    thumb_b = gr.Image(label="End", height=THUMB + 40,
                                       interactive=False,
                                       show_download_button=False)
                tslider = gr.Slider(0.0, 1.0, value=0.5, step=0.02,
                                    label="◀ Start  —  blend  —  End ▶")
                randomize = gr.Button("🎲  Randomize endpoints", variant="primary",
                                      size="lg")

                za_state = gr.State()
                zb_state = gr.State()

                randomize.click(
                    random_endpoints, [dataset],
                    [za_state, zb_state, thumb_a, thumb_b, blended, tslider])
                dataset.change(
                    on_dataset_change, [dataset],
                    [za_state, zb_state, thumb_a, thumb_b, blended, tslider])
                tslider.release(
                    on_blend, [dataset, za_state, zb_state, tslider], [blended])

        # Initialize both tabs on load.
        def _init():
            d_img, d_pad, d_lbl = digit_from_latent(0.0, 0.0)
            za, zb, ta, tb, bl, t = random_endpoints("digits")
            return d_img, d_pad, d_lbl, za, zb, ta, tb, bl, t

        demo.load(_init, None,
                  [out_digit, pad, coord_lbl,
                   za_state, zb_state, thumb_a, thumb_b, blended, tslider])

    return demo


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Launch the Latent Explorer booth UI")
    ap.add_argument("--share", action="store_true",
                    help="create a public gradio.live link (handy on remote "
                         "JupyterLab / cloud where localhost isn't reachable)")
    ap.add_argument("--port", type=int, default=7860, help="server port")
    ap.add_argument("--root-path", default=None,
                    help="mount path when behind a proxy, e.g. /proxy/7860 for "
                         "jupyter-server-proxy")
    cli = ap.parse_args()

    # Eagerly load both models so the booth is responsive from the first tap.
    for _n in ("digits", "galaxies"):
        try:
            get_model(_n)
            print(f"loaded model: {_n}")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
    build_ui().launch(server_name="0.0.0.0", server_port=cli.port,
                      share=cli.share, root_path=cli.root_path, show_api=False)
