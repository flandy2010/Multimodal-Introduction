#!/bin/bash
# ============================================================
# 3DGS 训练脚本 - 针对单张 NVIDIA H20 (96GB HBM3) 优化
# 修复版：解决光团问题
#   - scale 上限 0.008（防膨胀）
#   - opacity reset 每 3000 步（清理幽灵球）
#   - SSIM Loss（防模糊）
#   - grad_threshold 0.0004（控制分裂速度）
# ============================================================

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="../data/360_extra_scenes/flowers"
EXP_DIR="./runs/h20_flowers_v2"

python train.py \
    --data_path $DATA_PATH \
    --exp_dir $EXP_DIR \
    --factor 4 \
    --num_points 50000 \
    --n_iters 30000 \
    --lr 1e-2 \
    --sh_degree 3 \
    --tile_size 64 \
    --grad_threshold 0.0004 \
    --display_int 500 \
    --device cuda

echo "✅ 训练完成！结果在: $EXP_DIR"
