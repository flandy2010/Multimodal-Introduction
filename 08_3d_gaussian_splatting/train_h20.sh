#!/bin/bash
# ============================================================
# 3DGS H20 训练（性能优化版）
#
# 优化点：
#   1. tile_size=256（减少 tile 循环次数，H20 显存足够）
#   2. grad_threshold=0.0005（控制点数增长，防止变慢）
#   3. max_points=80000（硬上限）
#   4. factor=4（高分辨率 1256x828）
#   5. renderer 内使用展开公式计算高斯，避免矩阵乘法
# ============================================================

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="../data/360_extra_scenes/flowers"
EXP_DIR="./runs/h20_v3"

python train.py \
    --data_path $DATA_PATH \
    --exp_dir $EXP_DIR \
    --factor 4 \
    --num_points 50000 \
    --n_iters 30000 \
    --sh_degree 3 \
    --tile_size 256 \
    --grad_threshold 0.0005 \
    --display_int 500 \
    --device cuda

echo "Done! Results: $EXP_DIR"
