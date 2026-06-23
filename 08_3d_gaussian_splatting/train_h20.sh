#!/bin/bash
# ============================================================
# 3DGS 训练 - H20 (96GB) 论文标准配置
#
# 学习率（绝对值，不依赖 --lr 参数）：
#   means:    0.00016 → 0.0000016（指数衰减 100x）
#   opacity:  0.05（恒定）
#   sh_coeff: 0.0025（恒定）
#   scales:   0.005（恒定）
#   rotation: 0.001（恒定）
#
# 密度控制：
#   interval=100, threshold=0.0002, reset_opacity 每 3000 步
#   分裂/克隆按 scale 阈值区分, prune_opacity=0.005
# ============================================================

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="../data/360_extra_scenes/flowers"
EXP_DIR="./runs/h20_paper_config"

python train.py \
    --data_path $DATA_PATH \
    --exp_dir $EXP_DIR \
    --factor 4 \
    --num_points 50000 \
    --n_iters 30000 \
    --sh_degree 3 \
    --tile_size 128 \
    --grad_threshold 0.0002 \
    --display_int 500 \
    --device cuda

echo "Done! Results: $EXP_DIR"
