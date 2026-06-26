#!/bin/bash
# ============================================================
# 3DGS H20 训练
#
# 显存安全配置：
#   tile_size=64 → 单 tile 峰值 2000×4096×4B ≈ 32MB
#   factor=4 (1256x828) → ~320 tiles/帧
#   max_per_tile=2000（renderer 内硬编码）
#   如果仍 OOM，改 factor=8 (628x414) → ~70 tiles/帧
# ============================================================

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="../data/360_extra_scenes/flowers"
EXP_DIR="./runs/h20_v4"

rm -rf $EXP_DIR

python train.py \
    --data_path $DATA_PATH \
    --exp_dir $EXP_DIR \
    --factor 4 \
    --num_points 50000 \
    --max_points 1500000 \
    --n_iters 30000 \
    --sh_degree 3 \
    --tile_size 64 \
    --grad_threshold 0.0005 \
    --display_int 1000 \
    --device cuda

echo "Done! Results: $EXP_DIR"
