#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="${SCRIPT_DIR}/runs/h20_v5"

# 360° 轨道完全基于训练相机分布拟合：
#   - 圆心 / 半径 / 平面法线 = SVD 拟合训练相机
#   - look-at target = 所有训练视线的最小二乘交点（被拍主体）
#   - 内参直接复用训练相机焦距
# orbit_scale=1.0 时相机与训练同距；height_offset_ratio=0 时与训练分布同高
python "${SCRIPT_DIR}/inference.py" \
  --model_path "${EXP_DIR}/gs_final.pth" \
  --output_gif "${EXP_DIR}/walkthrough_orbit.gif" \
  --width 800 \
  --height 800 \
  --n_frames 120 \
  --fps 20 \
  --orbit_scale 1.0 \
  --height_offset_ratio 0.0 \
  --tile_size 32 \
  --device cuda
