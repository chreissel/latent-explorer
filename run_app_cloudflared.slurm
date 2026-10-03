#!/bin/bash
#SBATCH --job-name=latent-explorer
#SBATCH --partition=shared
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=latent_explorer_%j.out
#SBATCH --error=latent_explorer_%j.err
# #SBATCH --account=iaifi_lab           # set if your default fairshare account isn't right

set -uo pipefail

source /n/holystore01/LABS/iaifi_lab/Users/creissel/miniforge/etc/profile.d/conda.sh
conda activate latent-env
export LATENT_DATA_DIR=/n/holystore01/LABS/iaifi_lab/Lab/creissel/latent_explorer
cd /n/home03/creissel/latent-explorer

PORT=$(( 20000 + (RANDOM % 20000) ))

# --- get the cloudflared binary (public tunnel over ordinary outbound HTTPS) --
CF="$HOME/bin/cloudflared"
if [ ! -x "$CF" ]; then
  mkdir -p "$HOME/bin"
  echo "Downloading cloudflared ..."
  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
       -O "$CF" && chmod +x "$CF"
fi

# --- start the app locally (no gradio --share; cloudflared does the exposing) --
echo "Starting app on $(hostname):${PORT}"
python app.py --port "${PORT}" &
APP_PID=$!
trap 'kill ${APP_PID} 2>/dev/null' EXIT

# wait until the app is actually serving
for i in $(seq 1 90); do
  curl -s "http://localhost:${PORT}/" >/dev/null 2>&1 && break
  sleep 3
done

echo "==================================================================="
echo " PUBLIC URL: look for https://<something>.trycloudflare.com below"
echo "==================================================================="
# runs in the foreground for the rest of the walltime (48h)
"$CF" tunnel --no-autoupdate --url "http://localhost:${PORT}"
