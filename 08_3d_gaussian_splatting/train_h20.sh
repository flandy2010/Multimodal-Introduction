#!/bin/bash
# ============================================================
# 3DGS H20 训练
# ============================================================

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="../data/360_extra_scenes/flowers"
EXP_DIR="./runs/h20_v5"

rm -rf $EXP_DIR

python train.py \
    --data_path $DATA_PATH \
    --exp_dir $EXP_DIR \
    --factor 4 \
    --num_points 15000 \
    --max_points 1500000 \
    --n_iters 30000 \
    --sh_degree 3 \
    --tile_size 64 \
    --grad_threshold 0.0005 \
    --display_int 1000 \
    --device cuda \
    --scale_reg 0.01 \
    --opacity_reg 0.01 \
    --radius_clip 0.0

echo "Done! Results: $EXP_DIR"
