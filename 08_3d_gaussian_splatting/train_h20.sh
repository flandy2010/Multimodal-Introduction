#!/bin/bash
# ============================================================
# 3DGS 训练脚本 - 针对单张 NVIDIA H20 (96GB HBM3) 优化
# H20 特点：96GB 显存，FP32 性能约 60 TFLOPS
# ============================================================

# --- 环境配置 ---
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 训练参数 ---
# H20 有 96GB 显存，可以激进使用：
#   - factor=2 (全分辨率 2512x1656, 太大) 或 factor=4 (1256x828)
#   - 不限制渲染点数（让所有可见点参与）
#   - 更多迭代步数

DATA_PATH="../data/360_extra_scenes/flowers"
EXP_DIR="./runs/h20_flowers_sh3"

python train.py \
    --data_path $DATA_PATH \
    --exp_dir $EXP_DIR \
    --factor 4 \
    --num_points 50000 \
    --n_iters 30000 \
    --lr 1e-2 \
    --grad_threshold 0.0002 \
    --display_int 500 \
    --device cuda

echo "✅ 训练完成！结果在: $EXP_DIR"
