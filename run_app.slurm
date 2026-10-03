#!/bin/bash
#SBATCH --job-name=latent-explorer
#SBATCH --partition=shared              # CPU partition (allows multi-day jobs)
#SBATCH --time=48:00:00                 # 48-hour walltime
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=latent_explorer_%j.out
#SBATCH --error=latent_explorer_%j.err
# #SBATCH --account=iaifi_lab           # uncomment/set if your default fairshare account isn't right

set -uo pipefail

# --- activate the conda env -------------------------------------------------
source /n/holystore01/LABS/iaifi_lab/Users/creissel/miniforge/etc/profile.d/conda.sh
conda activate latent-env

# --- where the datasets live (harmless: baked maps usually make this unused) -
export LATENT_DATA_DIR=/n/holystore01/LABS/iaifi_lab/Lab/creissel/latent_explorer

cd /n/home03/creissel/latent-explorer

# --- pick a random high port so jobs sharing a node don't collide -----------
PORT=$(( 20000 + (RANDOM % 20000) ))
echo "Launching latent-explorer on node $(hostname) port ${PORT}"
echo "The public gradio.live URL will appear a few lines below."

# --share creates a public gradio.live link (lasts ~1 week; the job lasts 48h)
python app.py --share --port "${PORT}"
