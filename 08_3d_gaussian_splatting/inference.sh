#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="${SCRIPT_DIR}/runs/h20_v5"

python "${SCRIPT_DIR}/inference.py" \
  --model_path "${EXP_DIR}/gs_final.pth" \
  --output_gif "${EXP_DIR}/walkthrough_orbit.gif" \
  --width 800 \
  --height 800 \
  --fov_deg 60 \
  --n_frames 120 \
  --fps 20 \
  --orbit_scale 2.2 \
  --height_ratio 0.15 \
  --tile_size 32 \
  --device cuda
