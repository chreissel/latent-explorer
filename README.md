# 🌌 Latent Explorer

An interactive, touch-friendly demo for a science-fair booth. Students explore
the **latent space** of a small generative model and watch images **morph in
real time**. Once set up, it runs **fully offline on a single laptop**.

## The teaching point: one engine, two datasets

The **same code** (`vae.py`) trains variational autoencoders (VAEs). The *only*
differences are the dataset and the size of the latent space. The booth app uses
**Digits** and **Gravity Spy**:

| Model          | Dataset | Image | Latent space |
|----------------|---------|-------|--------------|
| **Digits**     | MNIST | 28×28 grayscale | **2-D** (fits on a flat pad) |
| **Gravity Spy**| LIGO glitch spectrograms | 64×64 (Q-transform), 22 classes | **32-D** (shown via PCA) |
| _Galaxies_     | Galaxy10 (SDSS) | 69×69 colour, 10 classes | 24-D _(still trainable; not shown by default)_ |

It's the identical machine — it just learned very different things: handwritten
digits vs. real gravitational-wave detector glitches.

## What's in the app (Gradio)

- **Image generation** — the latent space is drawn as a **class map** (colour
  clouds showing where each class lives; digits are labelled with numerals,
  spectrograms/galaxies with example thumbnails). **Tap anywhere** and the
  decoded image appears, upscaled. X/Y sliders mirror the tap.
- **Morphing** (both datasets) — pick two real samples and a slider blends
  smoothly from one to the other *through latent space*, with the **path drawn
  live on the map**. Includes a **🎲 Randomize endpoints** button.
- For datasets whose latent is bigger than 2-D (Gravity Spy 32-D, Galaxy10
  24-D), the map is a **PCA projection** to 2-D — a shadow of the full space.
- Gravity Spy spectrograms are colourised (viridis) for a booth-friendly look.

---

## Setup (one-time, needs internet)

Tested on Linux with Python 3.11. ~4 minutes of training total on CPU (seconds
on a GPU). Works CPU-only — a GPU is optional and only speeds up training.

```bash
# 1. Get the code
git clone https://github.com/chreissel/latent-explorer.git
cd latent-explorer
# (if it isn't on the default branch yet:)
# git checkout claude/optimistic-dirac-6p1kap

# 2. Create and activate a virtual environment
python3 -m venv latent-env      # 'latent-env' is the environment folder name — pick any name you like
source latent-env/bin/activate
pip install --upgrade pip
```

**3. Install PyTorch — pick ONE of the two:**

*CPU-only (small download, no GPU needed — recommended for a portable booth):*
```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu \
            --extra-index-url https://pypi.org/simple
```

*NVIDIA GPU (CUDA):*
```bash
nvidia-smi   # check the driver's CUDA version (top-right), then pick a matching tag below
#   cu121 = CUDA 12.1 | cu124 = CUDA 12.4 (good default) | cu128 = CUDA 12.8 (newest)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "gradio>=5,<6" numpy pillow h5py requests   # the rest of the deps
# verify (want: True + your GPU name):
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
```

```bash
# 4. Train the app's models (downloads MNIST + Gravity Spy, then trains -> models/)
python train.py            # digits + gravityspy; header prints 'on cpu' or 'on cuda'
```

> **Tip:** the CPU `--index-url …/whl/cpu` pulls the small CPU build of PyTorch.
> If you drop it, pip may install a much larger CUDA build (which still runs fine
> on CPU). No code changes are needed to switch — `train.py`/`app.py` auto-detect
> the GPU. If `nvidia-smi` works but the verify line prints `False`, your wheel's
> CUDA is newer than the driver — step down a tag (`cu124` → `cu121`) and reinstall.

### Training times (4-core CPU, no GPU)

| Step                    | Approx. time | Notes |
|-------------------------|--------------|-------|
| Digits (MNIST)          | ~1–2 min     | 20 epochs, 2-D latent |
| Gravity Spy (LIGO)      | ~3–6 min     | 40 epochs, 32-D latent; first run downloads a spectrogram `.h5` |
| _Galaxies (optional)_   | ~3–6 min     | `--dataset galaxies`; 40 epochs, 24-D latent; ~200 MB download |

Times vary a lot with CPU speed and core count — a recent laptop hits the
ranges above, while an older or throttled machine can take 3–5× longer. You
only train once.

VAE outputs are intentionally a little soft/blurry — that's expected for this
kind of model, and it makes the morphing look smooth.

#### Other datasets & offline stand-ins

```bash
python train.py --dataset galaxies      # also train the Galaxy10 model (still supported)
python train.py --dataset all           # digits + gravityspy + galaxies
```

If a download is blocked (restricted network) and you just want to see the
pipeline work, train on a built-in procedurally-generated stand-in:

```bash
python train.py --dataset gravityspy --synthetic   # or --dataset galaxies --synthetic
```

The stand-ins are fake (glitch-like blobs / fuzzy galaxies) — fine for a tech
demo, but **use the real data at the booth** (just run the normal command on a
connected machine once).

---

## Run the booth (works fully offline once trained)

```bash
source latent-env/bin/activate
python app.py
```

Then open **http://localhost:7860** in a browser (full-screen / kiosk mode is
nice for a booth). No internet required at this point — datasets are cached in
`data/` and weights are loaded from `models/`.

**Running on a remote host (e.g. JupyterLab / cloud)?** `localhost` won't be
reachable from your laptop. Either get a public link:

```bash
python app.py --share            # prints a https://xxxx.gradio.live URL (needs internet)
```

or, if `jupyter-server-proxy` is available, open
`https://<jupyter-host>/proxy/7860/` (with trailing slash) and launch with:

```bash
python app.py --root-path /proxy/7860
```

### Changing where datasets are stored

By default datasets download into `data/`. To put them elsewhere (e.g. a big
external disk), set the `LATENT_DATA_DIR` environment variable before running
`train.py` / `app.py` — no code changes needed:

```bash
export LATENT_DATA_DIR=/mnt/bigdisk/latent-data
python train.py        # datasets now download under $LATENT_DATA_DIR
```

---

## Project structure

```
latent-explorer/
├── vae.py             # The one engine: a configurable conv VAE (shared by both models)
├── data.py            # Dataset loading/caching (MNIST, Gravity Spy, Galaxy10, + stand-ins)
├── train.py           # Trains the models, saves weights to models/
├── app.py             # Loads weights and launches the Gradio booth UI
├── requirements.txt
└── README.md
```

Useful `train.py` flags:

```bash
python train.py --dataset digits            # just digits
python train.py --dataset gravityspy        # just Gravity Spy
python train.py --dataset galaxies          # just galaxies
python train.py --dataset all               # all three
python train.py --epochs 60                 # override epochs (train longer)
```

## Booth-day checklist

1. `source latent-env/bin/activate`
2. `python app.py`
3. Open `http://localhost:7860` full-screen.
4. No Wi-Fi needed — enjoy. 🎉
