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
from PIL import Image, ImageDraw, ImageFilter

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


# --- Projection to a clickable 2-D plane --------------------------------
# The digit latent is already 2-D; the galaxy latent is 24-D. To give BOTH a
# 2-D "map", we project each model's latent space down to 2-D: identity for
# digits, PCA (top-2 principal components) for galaxies. The map, the pad and
# the morph path all live in this projected plane. Moving in it changes the two
# most important latent directions while the rest stay at the data average.
def _encode_all(lm: LoadedModel, imgs: torch.Tensor, bs: int = 512) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, imgs.size(0), bs):
            mu, _ = lm.model.encode(imgs[i:i + bs].to(DEVICE))
            outs.append(mu.cpu().numpy())
    return np.concatenate(outs, axis=0)


def proj_to2d(m: dict, z) -> np.ndarray:
    return (np.asarray(z, dtype=np.float64) - m["mean"]) @ m["comp"].T


def proj_to_latent(m: dict, p) -> np.ndarray:
    return m["mean"] + np.asarray(p, dtype=np.float64) @ m["comp"]


def coord_to_px(x, y, xr, yr, W, H):
    return (x + xr) / (2 * xr) * W, (yr - y) / (2 * yr) * H


def px_to_coord(px, py, xr, yr, W, H):
    return px / W * (2 * xr) - xr, yr - py / H * (2 * yr)


def _clouds_from_points(P, labels, size, xr, yr, spread=0.6, fade=1.6):
    """Soft class-density image (no numerals) + de-collided label spots for a
    set of 2-D points P with integer class labels."""
    classes, means, invs = [], {}, {}
    reg = 1e-2 * ((xr + yr) / 2) ** 2
    for c in sorted({int(v) for v in labels}):
        Xc = P[labels == c]
        if len(Xc) < 5:
            continue
        cov = (np.cov(Xc.T) + np.eye(2) * reg) * spread
        means[c] = Xc.mean(0)
        invs[c] = np.linalg.inv(cov)
        classes.append(c)

    W = H = size
    xs = np.linspace(-xr, xr, W)
    ys = np.linspace(yr, -yr, H)
    gx, gy = np.meshgrid(xs, ys)
    G = np.stack([gx.ravel(), gy.ravel()], axis=1)

    Wc = np.empty((len(classes), G.shape[0]), dtype=np.float64)
    for i, c in enumerate(classes):
        d = G - means[c]
        Wc[i] = np.exp(-0.5 * np.einsum("pi,ij,pj->p", d, invs[c], d))

    colors = np.array([DIGIT_COLORS[c % len(DIGIT_COLORS)] for c in classes],
                      dtype=np.float64)
    wsum = Wc.sum(0) + 1e-8
    blended = (Wc.T @ colors) / wsum[:, None]
    alpha = np.clip(Wc.max(0), 0.0, 1.0) ** fade
    white = np.array([250.0, 250.0, 250.0])
    px_col = white[None, :] * (1 - alpha[:, None]) + blended * alpha[:, None]
    arr = np.clip(px_col, 0, 255).astype(np.uint8).reshape(H, W, 3)
    clouds = Image.fromarray(arr, mode="RGB")

    # Label positions (pixels), nudged apart so overlapping classes stay legible.
    pos = np.array([list(coord_to_px(means[c][0], means[c][1], xr, yr, W, H))
                    for c in classes], dtype=float)
    min_dist = 46.0
    for _ in range(120):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = pos[i] - pos[j]
                dist = float(np.hypot(d[0], d[1]))
                if dist < min_dist:
                    u = (d / dist) if dist > 1e-6 else np.array([1.0, 0.0])
                    shift = (min_dist - max(dist, 1e-6)) / 2.0
                    pos[i] += u * shift
                    pos[j] -= u * shift
                    moved = True
        if not moved:
            break
    pos[:, 0] = np.clip(pos[:, 0], 22, W - 22)
    pos[:, 1] = np.clip(pos[:, 1], 22, H - 22)
    label_spots = [(float(px), float(py), int(c))
                   for (px, py), c in zip(pos, classes)]
    return clouds, label_spots


def _draw_numerals(img: Image.Image, label_spots, size: int = 44) -> Image.Image:
    """Draw class numerals (crisply) on img: deep same-hue colour + light halo."""
    draw = ImageDraw.Draw(img)
    font = _load_font(size)
    for px, py, c in label_spots:
        deep = tuple(int(ch * 0.55) for ch in DIGIT_COLORS[c % len(DIGIT_COLORS)])
        for ox, oy in ((-2, -2), (2, -2), (-2, 2), (2, 2),
                       (-2, 0), (2, 0), (0, -2), (0, 2)):
            draw.text((px + ox, py + oy), str(c), fill=(255, 255, 255),
                      font=font, anchor="mm")
        draw.text((px, py), str(c), fill=deep, font=font, anchor="mm")
    return img


_MAP_CACHE: dict[str, dict] = {}


def get_map(lm: LoadedModel) -> dict:
    """Cached 2-D latent map for a model: projection + colour clouds + labels.
    Identity projection for a 2-D latent (digits); PCA for higher-D (galaxies).
    """
    if lm.name not in _MAP_CACHE:
        import data as datamod
        imgs, labels = datamod.load_labeled(lm.name, n=8000)
        Z = _encode_all(lm, imgs)
        labels = np.asarray(labels)
        if Z.shape[1] == 2:
            mean, comp = np.zeros(2), np.eye(2)
            xr = yr = LATENT_RANGE
            P = Z
        else:
            mean = Z.mean(0)
            Zc = Z - mean
            _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
            comp = Vt[:2]                       # top-2 principal directions
            P = Zc @ comp.T
            xr = float(np.percentile(np.abs(P[:, 0]), 98)) * 1.15 + 1e-6
            yr = float(np.percentile(np.abs(P[:, 1]), 98)) * 1.15 + 1e-6
        clouds, label_spots = _clouds_from_points(P, labels, 560, xr, yr)
        _MAP_CACHE[lm.name] = {"mean": mean, "comp": comp, "xr": xr, "yr": yr,
                               "clouds": clouds, "labels": label_spots}
    return _MAP_CACHE[lm.name]


_PAD_CACHE: dict[str, Image.Image] = {}


def get_pad_base(lm: LoadedModel) -> Image.Image:
    """Full latent map image: colour clouds with crisp numerals on top."""
    if lm.name not in _PAD_CACHE:
        m = get_map(lm)
        _PAD_CACHE[lm.name] = _draw_numerals(m["clouds"].copy(), m["labels"])
    return _PAD_CACHE[lm.name]


def pad_with_marker(lm: LoadedModel, x: float, y: float) -> Image.Image:
    """The latent map with a crosshair at projected coord (x, y)."""
    base = get_pad_base(lm).copy()
    m = get_map(lm)
    W, H = base.size
    px, py = coord_to_px(x, y, m["xr"], m["yr"], W, H)
    d = ImageDraw.Draw(base)
    r = 11
    d.ellipse([px - r, py - r, px + r, py + r], outline=(255, 255, 255), width=5)
    d.line([(px - r, py), (px + r, py)], fill=(255, 60, 60), width=3)
    d.line([(px, py - r), (px, py + r)], fill=(255, 60, 60), width=3)
    d.ellipse([px - r, py - r, px + r, py + r], outline=(255, 60, 60), width=3)
    return base


READOUT = ("<div style='font-size:1.7rem;font-weight:700;text-align:center'>"
           "input = ({:+.2f}, {:+.2f})</div>")


def generate_from_xy(dataset: str, x: float, y: float):
    """Decode the image at projected coord (x, y). Returns (image, pad, readout)."""
    lm = get_model(dataset)
    m = get_map(lm)
    z = proj_to_latent(m, [x, y])
    zt = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
    img = decode(lm, zt)
    return (tensor_to_pil(img, BIG, smooth=lm.channels == 3),
            pad_with_marker(lm, x, y),
            READOUT.format(x, y))


def on_pad_click(dataset: str, evt: gr.SelectData):
    lm = get_model(dataset)
    m = get_map(lm)
    W, H = get_pad_base(lm).size
    px, py = evt.index
    x, y = px_to_coord(px, py, m["xr"], m["yr"], W, H)
    out_img, pad_img, label = generate_from_xy(dataset, x, y)
    return out_img, pad_img, label, round(x, 2), round(y, 2)


def on_gen_dataset_change(dataset: str):
    """Switch the Image-generation tab to another dataset: reset pad + sliders
    to that model's 2-D map and coordinate ranges."""
    m = get_map(get_model(dataset))
    out_img, pad_img, label = generate_from_xy(dataset, 0.0, 0.0)
    return (out_img, pad_img, label,
            gr.update(minimum=-m["xr"], maximum=m["xr"], value=0.0),
            gr.update(minimum=-m["yr"], maximum=m["yr"], value=0.0))


# ---------------------------------------------------------------------------
# Morph / Interpolation (both datasets)
# ---------------------------------------------------------------------------
def interp_image(lm: LoadedModel, za: torch.Tensor, zb: torch.Tensor, t: float):
    z = (1 - t) * za + t * zb
    img = decode(lm, z.unsqueeze(0))
    return tensor_to_pil(img, BIG, smooth=lm.channels == 3)


def trajectory_map(dataset: str, za, zb, t: float):
    """Draw the morph path A->B on the model's 2-D latent map (PCA-projected for
    galaxies), with the current interpolation point marked."""
    if za is None or zb is None:
        return None
    lm = get_model(dataset)
    m = get_map(lm)
    base = m["clouds"].filter(ImageFilter.GaussianBlur(radius=2))
    base = Image.blend(base, Image.new("RGB", base.size, (255, 255, 255)), 0.15)
    _draw_numerals(base, m["labels"])
    W, H = base.size
    a2, b2 = proj_to2d(m, za.numpy()), proj_to2d(m, zb.numpy())
    c2 = (1 - t) * a2 + t * b2
    ax, ay = coord_to_px(a2[0], a2[1], m["xr"], m["yr"], W, H)
    bx, by = coord_to_px(b2[0], b2[1], m["xr"], m["yr"], W, H)
    cx, cy = coord_to_px(c2[0], c2[1], m["xr"], m["yr"], W, H)
    d = ImageDraw.Draw(base)
    d.line([(ax, ay), (bx, by)], fill=(10, 10, 10), width=10)
    d.line([(ax, ay), (bx, by)], fill=(230, 30, 30), width=4)
    for x, y in ((ax, ay), (bx, by)):
        r = 12
        d.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2],
                  fill=(255, 255, 255))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(10, 10, 10))
    r = 16
    d.ellipse([cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3],
              fill=(255, 255, 255))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(230, 30, 30),
              outline=(10, 10, 10), width=4)
    return base


def _fresh_endpoints(dataset: str):
    """New random endpoints + all Morphing-tab outputs (incl. trajectory map)."""
    lm = get_model(dataset)
    n = lm.sample_bank.size(0)
    ia, ib = np.random.choice(n, size=2, replace=False)
    za = encode_mu(lm, lm.sample_bank[ia])
    zb = encode_mu(lm, lm.sample_bank[ib])
    smooth = lm.channels == 3
    thumb_a = tensor_to_pil(lm.sample_bank[ia], THUMB, smooth)
    thumb_b = tensor_to_pil(lm.sample_bank[ib], THUMB, smooth)
    blended = interp_image(lm, za, zb, 0.5)
    traj = trajectory_map(dataset, za, zb, 0.5)
    return za, zb, thumb_a, thumb_b, blended, traj, 0.5


# random-endpoints button and dataset switch share the same behaviour.
random_endpoints = _fresh_endpoints
on_dataset_change = _fresh_endpoints


def on_blend(dataset: str, za, zb, t: float):
    lm = get_model(dataset)
    if za is None or zb is None:
        return None, None
    return interp_image(lm, za, zb, t), trajectory_map(dataset, za, zb, t)


def on_traj_click(dataset: str, za, zb, evt: gr.SelectData):
    """Click on the path plot -> slide the blend point to the nearest spot along
    the projected A->B line. Keeps the image, map and slider in sync."""
    if za is None or zb is None:
        return gr.update(), gr.update(), gr.update()
    lm = get_model(dataset)
    m = get_map(lm)
    W, H = m["clouds"].size
    px, py = evt.index
    x, y = px_to_coord(px, py, m["xr"], m["yr"], W, H)
    a2, b2 = proj_to2d(m, za.numpy()), proj_to2d(m, zb.numpy())
    ab = b2 - a2
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-9 else float((np.array([x, y]) - a2) @ ab / denom)
    t = min(max(t, 0.0), 1.0)
    return (interp_image(lm, za, zb, t),
            trajectory_map(dataset, za, zb, t),
            round(t, 2))


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
BIG_FONT_CSS = """
:root { --text-lg: 20px; --text-xl: 26px; }
.gradio-container, .gradio-container * { font-size: 1.12rem; }
button.svelte-1ixn6qd, .tab-nav button, button[role="tab"] {
    font-size: 1.5rem !important; font-weight: 700 !important;
    padding: 0.5em 1em !important;
}
.gradio-container h1 { font-size: 2.6rem !important; font-weight: 800; }
label span, .label-wrap span, span[data-testid] { font-size: 1.15rem !important; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Image generation", css=BIG_FONT_CSS,
                   theme=gr.themes.Soft(
                       font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
                       text_size=gr.themes.sizes.text_lg)) as demo:
        gr.Markdown("# Image generation")

        with gr.Tabs():
            # --- Image generation: click the 2-D latent map -----------
            with gr.Tab("✏️  Image generation"):
                gen_ds = gr.Radio(
                    choices=[("Digits", "digits"), ("Galaxies", "galaxies")],
                    value="digits", label="Dataset")
                with gr.Row(equal_height=False):
                    # Left column: input pad + sliders + readout (all share the
                    # pad's width, so the sliders match the left plot's length).
                    with gr.Column():
                        coord_lbl = gr.Markdown()
                        pad = gr.Image(label="Input — tap to explore",
                                       interactive=False, height=480,
                                       show_download_button=False)
                        sx = gr.Slider(-LATENT_RANGE, LATENT_RANGE, value=0.0,
                                       step=0.05, label="latent X")
                        sy = gr.Slider(-LATENT_RANGE, LATENT_RANGE, value=0.0,
                                       step=0.05, label="latent Y")
                    with gr.Column():
                        out_digit = gr.Image(label="Generated Image", height=480,
                                             show_download_button=False)

                pad.select(on_pad_click, [gen_ds],
                           [out_digit, pad, coord_lbl, sx, sy])
                for s in (sx, sy):
                    s.input(generate_from_xy, [gen_ds, sx, sy],
                            [out_digit, pad, coord_lbl], show_progress="hidden")
                gen_ds.change(on_gen_dataset_change, [gen_ds],
                              [out_digit, pad, coord_lbl, sx, sy])

            # --- Morphing: interpolate two samples along a path -------
            with gr.Tab("🔀  Morphing"):
                dataset = gr.Radio(
                    choices=[("Digits", "digits"), ("Galaxies", "galaxies")],
                    value="digits", label="Dataset")
                with gr.Row(equal_height=False):
                    # The morph path drawn on the (projected) latent map.
                    traj_map = gr.Image(label="Path through latent space",
                                        height=BIG, interactive=False,
                                        show_download_button=False)
                    with gr.Column():
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
                        randomize = gr.Button("🎲  Randomize endpoints",
                                              variant="primary", size="lg")

                za_state = gr.State()
                zb_state = gr.State()

                morph_out = [za_state, zb_state, thumb_a, thumb_b, blended,
                             traj_map, tslider]
                randomize.click(random_endpoints, [dataset], morph_out)
                dataset.change(on_dataset_change, [dataset], morph_out)
                # .input fires live while dragging, so the red dot slides along
                # the path in real time (not just on release).
                tslider.input(
                    on_blend, [dataset, za_state, zb_state, tslider],
                    [blended, traj_map], show_progress="hidden")
                # Click on the path plot to jump the dot to that point.
                traj_map.select(
                    on_traj_click, [dataset, za_state, zb_state],
                    [blended, traj_map, tslider], show_progress="hidden")

        # Initialize both tabs on load.
        def _init():
            d_img, d_pad, d_lbl = generate_from_xy("digits", 0.0, 0.0)
            za, zb, ta, tb, bl, traj, t = random_endpoints("digits")
            return d_img, d_pad, d_lbl, za, zb, ta, tb, bl, traj, t

        demo.load(_init, None,
                  [out_digit, pad, coord_lbl,
                   za_state, zb_state, thumb_a, thumb_b, blended,
                   traj_map, tslider])

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
