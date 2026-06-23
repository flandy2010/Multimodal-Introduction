#!/bin/bash
# ============================================================
# 3DGS 训练脚本 - 针对单张 NVIDIA H20 (96GB HBM3) 优化
# H20 特点：96GB 显存，FP32 性能约 60 TFLOPS
#
# 显存估算（tile-based renderer）：
#   - 模型参数: 50000 点 × (3+3+4+1+48) SH3 ≈ 12MB
#   - 单 tile 峰值: 4000 × 64×64 × 4B ≈ 63MB (可接受)
#   - 图像 + 梯度: 1256×828×3×2 ≈ 25MB
#   - 总峰值 < 5GB
# ============================================================

# --- 环境配置 ---
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="../data/360_extra_scenes/flowers"
EXP_DIR="./runs/h20_flowers_sh3"

python train.py \
    --data_path $DATA_PATH \
    --exp_dir $EXP_DIR \
    --factor 4 \
    --num_points 50000 \
    --n_iters 30000 \
    --lr 1e-2 \
    --sh_degree 3 \
    --tile_size 64 \
    --grad_threshold 0.0002 \
    --display_int 500 \
    --device cuda

echo "✅ 训练完成！结果在: $EXP_DIR"
